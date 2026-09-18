"""Local AutoDL transport: credentials stay in memory; known-host checks required."""
import argparse
import importlib.util
import json
from pathlib import Path
import types
import paramiko

def connect():
    script = Path.home() / '.codex/skills/autodl-instances/scripts/autodl.py'
    spec = importlib.util.spec_from_file_location('autodl', script)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    config = json.loads(api.CONFIG.read_text(encoding='utf-8-sig'))
    token = api.get_token(types.SimpleNamespace(env_file=None, token_key='AUTODL_API_KEY'), config)
    data = api.Client(token).call('GET', '/snapshot', {'instance_uuid': 'pro-78730289ac36'})['data']
    client = paramiko.SSHClient()
    client.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(data['proxy_host'], port=int(data['ssh_port']), username='root',
                   password=data['root_password'], look_for_keys=False, allow_agent=False, timeout=20)
    return client

def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['exec','put','get'])
    p.add_argument('source')
    p.add_argument('destination', nargs='?')
    args = p.parse_args()
    with connect() as client:
        if args.action == 'exec':
            _, stdout, stderr = client.exec_command(args.source)
            print(stdout.read().decode('utf-8',errors='replace'))
            print(stderr.read().decode('utf-8',errors='replace'))
            raise SystemExit(stdout.channel.recv_exit_status())
        with client.open_sftp() as sftp:
            if args.action == 'put': sftp.put(args.source,args.destination)
            else: sftp.get(args.source,args.destination)

if __name__ == '__main__': main()
