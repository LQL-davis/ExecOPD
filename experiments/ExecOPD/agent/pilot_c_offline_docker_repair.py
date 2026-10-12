"""Preserve all raw/failed trajectory evidence before infrastructure gates."""
import inspect
from experiments.ExecOPD.agent import pilot_c_offline as original
from experiments.ExecOPD.agent.pilot_docker_control_repair import install

source = inspect.getsource(original.collect)
old = """            gate(trajectory['state']['terminal_reason'] != 'SECURE_EXECUTOR_ERROR',
                 'secure Docker failed')
            append_sync(raw_path, {'event': 'trajectory_snapshot',"""
new = """            append_sync(raw_path, {'event': 'trajectory_snapshot',"""
original.gate(source.count(old) == 1, 'raw persistence insertion changed')
source = source.replace(old, new)
old = """            append_sync(path, trajectory)
            item, detail = inspect_item(tokenizer, trajectory)"""
new = """            append_sync(path, trajectory)
            gate(trajectory['state']['terminal_reason'] != 'SECURE_EXECUTOR_ERROR',
                 'secure Docker failed; full failed trajectory persisted')
            item, detail = inspect_item(tokenizer, trajectory)"""
original.gate(source.count(old) == 1, 'failed trajectory gate insertion changed')
source = source.replace(old, new)
old = """    finally:
        del model"""
new = """    finally:
        if not (folder / 'secure_tool_events.json').exists():
            write_new(folder / 'secure_tool_events.json', tool_events)
        del model"""
original.gate(source.count(old) == 1, 'tool evidence finalization changed')
namespace = dict(vars(original))
exec(source.replace(old, new), namespace)
original.collect = namespace['collect']

if __name__ == '__main__':
    install(original.ARTIFACTS / 'docker_control_audit.jsonl')
    original.main()
