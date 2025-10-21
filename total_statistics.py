import pandas as pd
from pathlib import Path
from openpyxl import load_workbook

ROOT_DIR = Path("source")

UPPER_ORDER = ["1-1", "2-1", "3-1", "4-1"]
LOWER_ORDER = ["1-2", "2-2", "3-2", "4-2"]

def extract_stats(xlsx_path):
    df = pd.read_excel(xlsx_path, header=None)
    # 첫 번째 row (0행)의 값들 반환
    first_row = df.iloc[0, :].dropna().tolist()
    # 두 번째 row (1행)의 값들 반환
    second_row = df.iloc[1, :len(first_row)].tolist()
    return first_row, second_row

def build_wide_table(order_list, xlsx_files):
    # 통계 항목 개수 파악 (첫 번째 파일에서)
    first_key = next((k for k in order_list if k in xlsx_files), None)
    if first_key:
        first_row, _ = extract_stats(xlsx_files[first_key])
        num_stats = len(first_row)
    else:
        num_stats = 14  # 기본값
        print(f"   ⚠️  파일을 찾을 수 없어 기본값 사용: {num_stats}")
    
    # Row 1: input 엑셀의 첫 번째 row
    first_data_row = ["종속변수"]
    for stat_idx in range(num_stats):
        for key in order_list:
            if key in xlsx_files:
                first_row, _ = extract_stats(xlsx_files[key])
                first_data_row.append(first_row[stat_idx])
            else:
                first_data_row.append("")
        
    # Row 2: 실험 조건 (1-1, 2-1, 3-1, 4-1 반복)
    condition_row = [""] + order_list * num_stats
    
    # Row 3: input 엑셀의 두 번째 row
    second_data_row = [""]
    for stat_idx in range(num_stats):
        for key in order_list:
            if key in xlsx_files:
                _, second_row = extract_stats(xlsx_files[key])
                second_data_row.append(second_row[stat_idx])
            else:
                second_data_row.append("")    
    df = pd.DataFrame([first_data_row, condition_row, second_data_row])
    return df, num_stats

def apply_merges(worksheet, start_row, num_stats):
    # 각 통계 항목 헤더 병합 (4칸씩)
    for i in range(num_stats):
        start_col = 2 + i * 4
        end_col = start_col + 3
        worksheet.merge_cells(start_row=start_row, start_column=start_col,
                            end_row=start_row, end_column=end_col)

def process_subject(subject_path):
    subject_name = subject_path.name
    xlsx_files = {f.parent.name: f for f in subject_path.rglob("_speed_time.xlsx")}
    print(f"\n📌 처리 중: {subject_name} (발견된 파일 {len(xlsx_files)}개)")
    print(f"   발견된 키: {list(xlsx_files.keys())}")

    print(f"\n🔼 상단 테이블 (UPPER) 생성 중...")
    upper_df, num_stats = build_wide_table(UPPER_ORDER, xlsx_files)
    
    print(f"\n🔽 하단 테이블 (LOWER) 생성 중...")
    lower_df, _ = build_wide_table(LOWER_ORDER, xlsx_files)

    output_path = subject_path / "total_speed_statistics.xlsx"
    
    # 데이터 쓰기
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        upper_df.to_excel(writer, sheet_name="통계", index=False, header=False, startrow=0)
        lower_df.to_excel(writer, sheet_name="통계", index=False, header=False, 
                         startrow=len(upper_df) + 2)
    
    # 셀 병합 적용
    wb = load_workbook(output_path)
    ws = wb["통계"]
    
    apply_merges(ws, start_row=1, num_stats=num_stats)  # 상단 테이블
    apply_merges(ws, start_row=len(upper_df) + 3, num_stats=num_stats)  # 하단 테이블
    
    wb.save(output_path)
    print(f"✅ 완료: {output_path}\n")

for subject_dir in ROOT_DIR.iterdir():
    if subject_dir.is_dir():
        process_subject(subject_dir)