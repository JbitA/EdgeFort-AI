from concurrent.futures import ThreadPoolExecutor
from edgeai.telemetry import OfflineSpool

def test_spool_concurrent_append(tmp_path):
 s=OfflineSpool(tmp_path/'s',100000)
 with ThreadPoolExecutor(max_workers=8) as ex:list(ex.map(lambda i:s.append({'i':i}),range(500)))
 got=[];assert s.flush(got.append)==500;assert len({x['i'] for x in got})==500
