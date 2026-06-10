# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Merged vand test case.

#!/usr/bin/python3
import os,sys
import numpy as np

def _cmp(golden,output,dtype,eps,count=-1):
    if not os.path.exists(golden) or not os.path.exists(output): return False
    kw={} if count<0 else {"count":count}
    g=np.fromfile(golden,dtype=dtype,**kw)
    o=np.fromfile(output,dtype=dtype,**kw)
    return g.shape==o.shape and np.allclose(g,o,atol=eps,rtol=eps,equal_nan=True)

def _cmpeq(golden,output,dtype):
    if not os.path.exists(golden) or not os.path.exists(output): return False
    g=np.fromfile(golden,dtype=dtype)
    o=np.fromfile(output,dtype=dtype)
    return g.shape==o.shape and np.array_equal(g,o)

def main():
    strict=os.getenv('COMPARE_STRICT','1')!='0'
    failed=[]
    if not (_cmp("golden_v3.bin","v3.bin",np.uint16,0,1024)):
        failed.append('f32')
        print('[ERROR] compare failed: f32')
    else:
        print('[INFO] f32: passed')
    if not (_cmpeq("golden_v3_mask_edge.bin","v3_mask_edge.bin",np.uint16)):
        failed.append('mask_edge')
        print('[ERROR] compare failed: mask_edge')
    else:
        print('[INFO] mask_edge: passed')
    if failed:
        if strict: print(f"[ERROR] {len(failed)} variant(s) failed"); sys.exit(2)
        print(f"[WARN] {len(failed)} variant(s) failed (non-gating)")
        return
    print("[INFO] compare passed (all 2 variants)")

if __name__=="__main__":
    main()
