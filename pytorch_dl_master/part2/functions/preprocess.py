from collections import namedtuple
import functools, glob, os, csv, sys
sys.path.append(os.getcwd())

from functions.util import IrcTuple, XyzTuple, irc2xyz, xyz2irc
from functions.util import logging
from functions.util import getCache
import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data import Dataset
import copy

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz')



@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool = True): # requireOnDisk_bool은 디스크에 없는 데이터 걸러내는 용도
    BASE_DIR = os.getcwd()
    pattern = os.path.join(BASE_DIR, "luna", "subset*", "*.mhd")
    mhd_list = glob.glob(pattern)
    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}
    # print(presentOnDisk_set)
    # presentOnDisk_list = list(presentOnDisk_set)
    # print(len(presentOnDisk_list))
    diameter_dict: dict[str, list[tuple[tuple[float, float, float], float]]] = {}
    with open(os.path.join(BASE_DIR, "luna", 'annotations.csv'), "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])
            
            # diameter_dict.setdefault(series_uid, []).append(
            #     (annotationCenter_xyz, annotationDiameter_mm)
            # )
            if series_uid not in diameter_dict:
                diameter_dict[series_uid] = []
            diameter_dict[series_uid].append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )
            
    candidateInfo_list = []
    with open(os.path.join(BASE_DIR, "luna", "candidates.csv"), 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            
            # series_uid가 없으면 서브셋에 있지만(candidates.csv) 디스크에는 없으므로 건너뛴다
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue
            
            isNodule_bool = bool(int(row[4])) 
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])
            
            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    # 반경을 얻기 위해 직경을 2로 나누고, 두 개의 결절 센터가 결절의 크기 기준으로
                    # 너무 떨어져 있는지를 반지름의 절반 길이를 기준으로 판정한다(실거리가 아닌 바운딩 박스 체크)
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break
            candidateInfo_list.append(CandidateInfoTuple(isNodule_bool, candidateDiameter_mm, series_uid, candidateCenter_xyz))
    
    # 모든 결절 샘플을 내림차순으로 정렬, 그 뒤에는 (크기 정보가 없는) 결절이 아닌 샘플이 이어져 있는 데이터가 준비되어있다.        
    candidateInfo_list.sort(reverse = True)
    return candidateInfo_list





class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob(
            os.path.join("luna", "subset*", f"{series_uid}.mhd")
        )[0]
        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        ct_a.clip(-1000, 1000, ct_a)
        
        self.series_uid = series_uid
        self.hu_a = ct_a
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc
    

raw_cache = getCache('part2ch10_raw')

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc


class LunaDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
            ):
        self.candidateInfo_list = copy.copy(getCandidateInfoList())

        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        log.info("{!r}: {} {} samples".format(
            self,
            len(self.candidateInfo_list),
            "validation" if isValSet_bool else "training",
        ))

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        candidateInfo_tup : CandidateInfoTuple = self.candidateInfo_list[ndx]
        width_irc = (32, 48, 48)

        candidate_a, center_irc = getCtRawCandidate(
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        # 결절의 원핫인코딩 표현 -> 결절이면 [0,1] , 결절 아니면 [1,0]
        pos_t = torch.tensor([
                not candidateInfo_tup.isNodule_bool, 
                candidateInfo_tup.isNodule_bool
            ],
            dtype=torch.long,
        )

        return (
            candidate_t,
            pos_t,
            candidateInfo_tup.series_uid,
            torch.tensor(center_irc),
        )