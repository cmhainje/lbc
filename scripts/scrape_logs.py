"""
scrape_logs.py
Connor Hainje (connor.hainje@nyu.edu)

Scrapes a log file from the sloan-25mlog mailing list for the frame numbers of
every BOSS arc/flat. Note: requires the user to be a member of the mailing
list. Subscribe here:
    https://mailman.sdss.org/mailman/listinfo/sloan-25mlog
"""

import argparse
import requests
import re
import json

from tqdm import tqdm


ap = argparse.ArgumentParser()
ap.add_argument('username')
ap.add_argument('password')
ap.add_argument('-f', '--filename', default='2023-December.txt')
ap.add_argument('-o', '--output', default='frames.json')
args = ap.parse_args()


BASE_URL = 'https://mailman.sdss.org/mailman/private/sloan-25mlog'

# First we send a request to the list URL to log in
r = requests.post(BASE_URL,
                  data={'username': args.username, 'password': args.password})

# If authentication is successful, we get back some cookies
cookie = r.cookies
if len(cookie) == 0:
    raise RuntimeError('Authentication failed.')
print('Authenticated!')

# Now we can request an auth-locked page...
r = requests.get(f"{BASE_URL}/{args.filename}", cookies=cookie)
if r.status_code != 200:
    raise RuntimeError(
        f'Request failed.\n'
        f'  Status code: {r.status_code}\n'
        f'  Reason: {r.reason}\n'
        f'  Text: {r.text}'
    )
print('Log file downloaded.')

# ...and scrape it for BOSS arcs and flats
# First we look for the BOSS data summary tables
start = r'---- BOSS Data Summary ----'  # this goes before the table
stop = r'---- ([^^\n]+?) ----'  # just look for the next header
pattern = re.compile(r'^(?<!>)' + start + r'(.*?)' + stop,
                     re.DOTALL | re.MULTILINE)  # made with ChatGPT
matches = pattern.findall(r.text)

arcs, flats = {}, {}

if not matches:
    raise RuntimeError('No BOSS summary tables found!')

print(f'{len(matches)} BOSS summary tables found. Parsing...')

for match in tqdm(matches, desc='Parsing', unit='table'):
    lines = match[0].strip().split('\n')

    # There are two completely different tables that I've seen appear in
    # the BOSS data summary. Assuming they're the only two...

    if lines[0].startswith('Reading FITS'):
        message = lines[0]
        header = lines[2]
        table = lines[4:]

        mjd = message.split('/')[-1]

        arc_frames, flat_frames = set(), set()
        for line in table:
            chunks = line.strip().split()
            try:
                frame_num = chunks[2].split('-')[1]
                flav = chunks[5]
            except:
                print(chunks)
                print('---')
                print(match)
                raise RuntimeError('borked!')
            if flav == 'arc':
                arc_frames.add(frame_num)
            elif flav == 'fla':
                flat_frames.add(frame_num)

        arcs[mjd] = sorted(list(arc_frames))
        flats[mjd] = sorted(list(flat_frames))

    else:
        if (
            lines[0].startswith('====')
            and lines[1].strip().startswith('BOSS')
            and lines[2].startswith('====')
        ):
            lines = lines[4:]

        header = lines[0]
        table = lines[2:]

        mjd = table[0].split()[0]

        arc_frames, flat_frames = set(), set()
        for line in table:
            try:
                exp, flav, _, _, _ = line.split()[-5:]
            except:
                print(line)
                print(match[0])
                raise RuntimeError('borked!')
            if flav == 'Arc':
                arc_frames.add(exp)
            elif flav == 'Flat':
                flat_frames.add(exp)

        arcs[mjd] = sorted(list(arc_frames))
        flats[mjd] = sorted(list(flat_frames))

print('Parsing complete!')

with open(args.output, 'w') as f:
    json.dump({'arcs': arcs, 'flats': flats}, f, indent=2)

print(f'The data has been written to {args.output}')
print('Done! Goodbye.')
