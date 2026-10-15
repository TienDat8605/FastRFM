"""Recover one allocation whose create response timed out, via CLI-native APIs.

Run with the Python interpreter belonging to google-colab-cli. Never print
proxy credentials or adopt a runtime already named in the local session store.
"""
import argparse
import json
from pathlib import Path
from colab_cli.common import state
from colab_cli.state import SessionState

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--session', required=True)
parser.add_argument('--snapshot', type=Path)
parser.add_argument('--before', type=Path)
args = parser.parse_args()
assignments = state.client.list_assignments()
if args.snapshot:
    args.snapshot.write_text(json.dumps([a.endpoint for a in assignments]))
    print('Saved pre-allocation endpoints')
    raise SystemExit(0)
known = {session.endpoint for session in state.store.list().values()}
if args.before:
    known.update(json.loads(args.before.read_text()))
available = [a for a in assignments
             if a.endpoint not in known and a.accelerator.value.lower() == 't4']
if len(available) != 1:
    raise SystemExit(f'Recovery requires exactly one unnamed T4 allocation; found {len(available)}')
runtime = available[0]
info = runtime.runtime_proxy_info
state.store.add(SessionState(name=args.session, endpoint=runtime.endpoint,
    url=info.url, token=info.token, token_expires_at=info.expires_at(),
    variant=runtime.variant.name, accelerator=runtime.accelerator.value,
    machine_shape=runtime.machine_shape.name))
state.history.log_event(args.session, 'session_recovered', {'endpoint': runtime.endpoint})
print(f'Recovered {args.session}: {runtime.endpoint}')
