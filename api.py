from __future__ import absolute_import, division, print_function

import os
import shlex
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_file, after_this_request
from stt import Model, version
from werkzeug.utils import secure_filename

from config import BEARER, PY_PATH

app = Flask(__name__)

app.config["DEBUG"] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

AUDIO_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "aiff", "wma", "m4a"}
MAX_AUDIO_LENGTH = 180.0

try:
    from shlex import quote
except ImportError:
    from pipes import quote


def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = "sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - ".format(
        quote(audio_path), desired_sample_rate
    )
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("SoX returned non-zero status: {}".format(e.stderr))
    except OSError as e:
        raise OSError(
            e.errno,
            "SoX not found, use {}hz files or install it: {}".format(
                desired_sample_rate, e.strerror
            ),
        )

    return desired_sample_rate, np.frombuffer(output, np.int16)


def metadata_to_string(metadata):
    return "".join(token.text for token in metadata.tokens)


def words_from_candidate_transcript(metadata):
    word = ""
    word_list = []
    word_start_time = 0
    # Loop through each character
    for i, token in enumerate(metadata.tokens):
        # Append character to word if it's not a space
        if token.text != " ":
            if len(word) == 0:
                # Log the start time of the new word
                word_start_time = token.start_time

            word = word + token.text
        # Word boundary is either a space or the last character in the array
        if token.text == " " or i == len(metadata.tokens) - 1:
            word_duration = token.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            each_word = dict()
            each_word["word"] = word
            each_word["start_time"] = round(word_start_time, 4)
            each_word["duration"] = round(word_duration, 4)

            word_list.append(each_word)
            # Reset
            word = ""
            word_start_time = 0

    return word_list


def metadata_json_output(metadata):
    json_result = dict()
    json_result["transcripts"] = [
        {
            "confidence": transcript.confidence,
            "words": words_from_candidate_transcript(transcript),
        }
        for transcript in metadata.transcripts
    ]
    return json_result



def allowed_file(filename):
    return '.' in filename and \
           filename.split('.')[-1].lower() in AUDIO_EXTENSIONS

def get_audio_length(audio_path):
    audio_path = shlex.quote(audio_path)
    return float(subprocess.check_output(f"ffprobe -i {audio_path} -show_entries format=duration -v quiet -of csv=\"p=0\"", shell=True).decode().strip())


def is_auth(request):
    return BEARER == request.headers.get("Authorization")

@app.route("/stt/languages", methods=["GET"])
def langs():
    if not is_auth(request):
        return jsonify({"error": "Unauthorized"}), 401
    languages = [p.name for p in Path("./models/").glob("*") if p.is_dir()]
    return jsonify(languages)

@app.route("/version", methods=["GET"])
def ver():
    if not is_auth(request):
        return jsonify({"error": "Unauthorized"}), 401
    return version()

@app.route("/stt/<lang>", methods=["POST"])
def stt_lib(lang):
    if not is_auth(request):
        return jsonify({"error": "Unauthorized"}), 401
    languages = [p.name for p in Path("./models/").glob("*") if p.is_dir()]
    if lang not in languages:
        return jsonify({"error": "Language not supported"})

    args = request.args
    file = None
    try:
        file = request.files.values().__next__()
    except StopIteration:
        pass

    # Download the file
    if file is not None:
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Filename not allowed " + file.filename})
        filename = secure_filename(file.filename)
        suffix = "." + filename.split(".")[-1]
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        file.save(audio_path.name)
        print("Downloaded file: {}".format(audio_path.name))

    elif "url" in args:
        url = args["url"]
        filename = url.split("/")[-1]
        if not allowed_file(filename):
            return jsonify({"error": "Filename not allowed " + filename})
        suffix = "." + filename.split(".")[-1]
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            if subprocess.call(["curl", url, "--max-filesize", str(app.config['MAX_CONTENT_LENGTH']), "--output", audio_path.name]):
                return jsonify({"error": "Failed to download file"})

        except Exception as e:
            return jsonify({"error": "Failed to download file: " + str(e)})
        print("Downloaded file: {}".format(audio_path.name))

    else:
        return jsonify({"error": "No url or file provided"})

    # Check audio length
    try:
        if get_audio_length(audio_path.name) > MAX_AUDIO_LENGTH:
            return jsonify({"error": "Audio file too long. Max length is {} seconds".format(MAX_AUDIO_LENGTH)})
    except Exception as e:
        os.remove(audio_path.name)
        return jsonify({"error": "Failed to get audio length: " + str(e)})

    # convert to wav with ffmpeg and if fails return error
    converted_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        if subprocess.call(["ffmpeg", "-y", "-i", audio_path.name, "-f", "wav", converted_audio_path.name]):
            return jsonify({"error": "Failed to convert audio file"})
    except subprocess.CalledProcessError as e:
        os.remove(audio_path.name)
        return jsonify({"error": "Failed to convert audio file " + str(e)})

    os.remove(audio_path.name)

    ds = Model("models/" + lang + "/model.tflite")
    desired_sample_rate = ds.sampleRate()
    ds.enableExternalScorer(str(Path("models/" + lang + "/").glob("*.scorer").__next__()))

    fin = wave.open(converted_audio_path.name, "rb")
    fs_orig = fin.getframerate()
    if fs_orig != desired_sample_rate:
        print(
            "Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.".format(
                fs_orig, desired_sample_rate
            ),
            file=sys.stderr,
        )
        fs_new, audio = convert_samplerate(converted_audio_path.name, desired_sample_rate)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    # remove files
    os.remove(converted_audio_path.name)
    _stt = ds.sttWithMetadata(audio, 3)
    transcript = metadata_json_output(_stt)
    transcript.update({"full": metadata_to_string(_stt.transcripts[0])})
    return jsonify(transcript)


TTS_LANGUAGES = {"de", "en", "es", "fr", "it", "nl", "ru", "sv", "sw"}
MAX_TTS_TEXT_LENGTH = 512

@app.route("/tts/languages", methods=["GET"])
def ttslangs():
    if not is_auth(request):
        return jsonify({"error": "Unauthorized"}), 401
    languages = list(TTS_LANGUAGES)
    return jsonify(languages)

@app.route("/tts/<lang>", methods=["POST"])
def tts(lang):
    if not is_auth(request):
        return jsonify({"error": "Unauthorized"}), 401
    if lang not in TTS_LANGUAGES:
        return jsonify({"error": "Language not supported"})
    text = request.form.get("text")
    if not text:
        return jsonify({"error": "No text provided"})
    if len(text) > MAX_TTS_TEXT_LENGTH:
        return jsonify({"error": "Text too long. Max length is {} characters".format(MAX_TTS_TEXT_LENGTH)})

    text = shlex.quote(text)
    filename = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    try:
        if subprocess.call(f"{PY_PATH} -m larynx {lang} \"{text}\" > {filename}", shell=True):
            return jsonify({"error": "Failed to generate audio file"})
    except subprocess.CalledProcessError as e:
        os.remove(filename)
        return jsonify({"error": "Failed to generate audio file " + str(e)})

    @after_this_request
    def remove_file(response):
        os.remove(filename)
        return response
    return send_file(filename, mimetype="audio/wav")

def main():
    app.run(host='127.0.0.1', port=5555)

if __name__ == "__main__":
    main()
