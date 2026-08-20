import json,pytest
from edgeai.telemetry import OfflineSpool

def test_append_flush(tmp_path):
 s=OfflineSpool(tmp_path/'x',1024);s.append({'a':1});out=[];assert s.flush(out.append)==1;assert out==[{'a':1}]
def test_failed_sink_remains(tmp_path):
 s=OfflineSpool(tmp_path/'x',1024);s.append({'a':1});assert s.flush(lambda _:(_ for _ in ()).throw(RuntimeError()))==0;assert s.stats()['bytes']>0
def test_corrupt_line_is_quarantined_by_counter(tmp_path):
 p=tmp_path/'x';p.write_bytes(b'{bad}\n');s=OfflineSpool(p,1024);assert s.flush(lambda _:None)==0;assert s.stats()['corrupt']==1
def test_bounded_spool_drops_oldest(tmp_path):
 s=OfflineSpool(tmp_path/'x',1024)
 for i in range(50):s.append({'i':i,'v':'x'*30})
 assert s.stats()['bytes']<=1024 and s.dropped>0
def test_oversize_event_dropped(tmp_path):
 s=OfflineSpool(tmp_path/'x',1024);assert not s.append({'x':'z'*2000})

def test_flush_preserves_order_after_sink_failure(tmp_path):
 s=OfflineSpool(tmp_path/'x',2048)
 for i in range(4):s.append({'i':i})
 seen=[]
 def sink(event):
  if event['i']==1:raise RuntimeError('offline')
  seen.append(event['i'])
 assert s.flush(sink)==1
 assert seen==[0]
 replay=[]
 assert s.flush(replay.append)==3
 assert [e['i'] for e in replay]==[1,2,3]


def test_flush_max_events_and_invalid_limit(tmp_path):
 s=OfflineSpool(tmp_path/'x',2048)
 for i in range(3):s.append({'i':i})
 out=[];assert s.flush(out.append,max_events=1)==1
 assert [e['i'] for e in out]==[0]
 out=[];assert s.flush(out.append)==2
 assert [e['i'] for e in out]==[1,2]
 with pytest.raises(ValueError):s.flush(out.append,max_events=-1)
