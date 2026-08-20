from edgeai.evaluation import robustness_campaign

def test_campaign_quality():
 r=robustness_campaign(range(3))['summary']
 assert r['weight_int8_clean_mean']>.95 and r['dynamic_int8_noise_016_mean']>.88
