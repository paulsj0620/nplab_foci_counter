# foci_counter

H&E 염색 CZI (Zeiss AxioScan) 간 조직 이미지에서 **inflammation foci** 개수를
자동으로 세는 도구. 참조 방법: Socha & Shumbayawonda 2024 (whole-slide 5단계
파이프라인), StarDist 사전학습 H&E 모델 사용.

## 설치

```bash
conda create -n focicnt python=3.12 -y
conda activate focicnt
pip install -r requirements.txt
```

## 데이터

`dataset/`의 원본 CZI(수십 GB)는 GitHub에 올리지 않습니다(`.gitignore`).
새 머신에서는 데이터셋을 별도로 복사하세요.

## 사용법 (2단계)

```bash
# 1) 핵 검출 — 슬라이드당 ~8분(Mac mini). 결과를 results/<stem>_nuclei.npz 로 캐시.
python scripts/detect_slide_nuclei.py dataset/<slide>.czi

# 2) 분석 + 세트 출력 — 캐시 기반, ~15초. 파라미터 튜닝 시 이 단계만 반복.
python scripts/run_slide.py results/<stem>_nuclei.npz dataset/<slide>.czi
```

무거운 딥러닝(1단계)은 슬라이드당 한 번만, 이후 튜닝(2단계)은 캐시로 빠르게 반복하도록
분리했습니다.

## 산출물 (`results/<slide>/`)

- `<slide>_overview.png` — 전체 슬라이드 지도 (foci = 빨간 링, 검토 ROI = 노란 번호 사각형)
- `<slide>_gallery.png` — ROI별 확대 크롭 (foci를 큰 노란 원으로 표시)
- `<slide>_results.xlsx` — summary(foci 수, FD/mm²·/µm², 조직면적, ROI 커버리지) + ROI별 시트

번호가 지도 ↔ 갤러리 ↔ 엑셀에서 서로 연결됩니다.

## 파이프라인 모듈 (`scripts/`)

| 모듈 | 역할 |
|---|---|
| `czi_loader.py` | CZI 피라미드 로드, 다운샘플/영역 읽기 |
| `tissue_mask.py` | 조직/배경 분리 (찢김 제외) |
| `tissue_pieces.py` | 조직 조각 분리 → 가장 깨끗한 조각 선택 |
| `tiling.py` | 조직 타일 격자 생성 |
| `nuclei.py` | StarDist 핵 검출 |
| `inflammation.py` | 염증세포 판별 (크기 + 밀도) |
| `foci.py` | foci 군집화 (DBSCAN) |
| `foci_pipeline.py` | 4-5단계 + QC 필터 (조직비율/밝기/채도/혈관) |
| `roi_select.py` | 대표 ROI 선택 (비겹침, 조직 커버리지) |
| `detect_slide_nuclei.py` | 1단계 배치 (핵 검출 캐시) |
| `run_slide.py` | 2단계 배치 (분석 + 세트 출력) |
