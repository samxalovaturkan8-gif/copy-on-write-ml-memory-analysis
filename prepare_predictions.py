import json, math, os
from pathlib import Path
configs=[('E1',4,4,.25,1024),('E2',4,4,0,1024),('E3',4,4,.5,1024),('E4',4,4,1,1024),('E5',8,8,.25,1024),('E6',4,4,.25,4096),('E7',4,2,.5,1024),('E8',4,4,.75,1024)]
page=os.sysconf('SC_PAGE_SIZE'); rows=[]
for name,w,wm,p,h in configs:
    params=h*h+76*h+10
    pages=math.ceil(params*4/page); k=round(p*pages); s=pages*page/2**20
    rows.append(dict(experiment=name,workers=w,modify_workers=wm,modify=p,width=h,epochs=10,repeats=5,page_bytes=page,model_pages=pages,planned_writes=wm*k,independent_model_pss_mib=w*s,cow_initial_model_pss_mib=s,cow_modified_model_pss_mib=s+wm*k*page/2**20,independent_model_rss_sum_mib=w*s,cow_model_rss_sum_mib=(w+1)*s))
Path('predictions.json').write_text(json.dumps(dict(scope='Ideal resident model mapping only; whole-process overhead is not predicted.', configurations=rows),indent=2)+'\n')
