#!/usr/bin/env python3
"""Submit a WAV file to a local Tongues server using only the standard library."""

import argparse
import json
import urllib.parse
import urllib.request


parser = argparse.ArgumentParser()
parser.add_argument("wav")
parser.add_argument("--server", default="http://127.0.0.1:3000")
parser.add_argument("--provider", default="whisper.cpp")
parser.add_argument("--language", default="en")
args = parser.parse_args()

query = urllib.parse.urlencode(
    {"provider": args.provider, "language": args.language}
)
url = f"{args.server.rstrip('/')}/api/asr/transcriptions?{query}"
with open(args.wav, "rb") as wav:
    request = urllib.request.Request(
        url, data=wav.read(), headers={"Content-Type": "audio/wav"}, method="POST"
    )
with urllib.request.urlopen(request) as response:
    result = json.load(response)
print(result["transcript"])
