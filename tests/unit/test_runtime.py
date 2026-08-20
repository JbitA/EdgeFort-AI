import time,pytest
from edgeai.runtime import BoundedInferenceQueue,OverflowPolicy
from edgeai.errors import QueueFull

def test_drop_oldest():
 q=BoundedInferenceQueue(2);q.push(1);q.push(2);q.push(3);assert q.pop().payload==2 and q.stats()['dropped']==1
def test_drop_newest():
 q=BoundedInferenceQueue(1,OverflowPolicy.DROP_NEWEST);assert q.push(1);assert not q.push(2);assert q.pop().payload==1
def test_reject():
 q=BoundedInferenceQueue(1,OverflowPolicy.REJECT);q.push(1)
 with pytest.raises(QueueFull):q.push(2)
def test_expired_items():
 q=BoundedInferenceQueue(2);q.push(1,deadline_s=0);time.sleep(.002);assert q.pop() is None and q.stats()['expired']==1
def test_invalid_capacity():
 with pytest.raises(ValueError):BoundedInferenceQueue(0)

def test_invalid_deadline_rejected():
 q=BoundedInferenceQueue(1)
 for value in (-1, float('inf'), float('nan')):
  with pytest.raises(ValueError):q.push(1, deadline_s=value)
