"""새로운 데이터셋 확인 스크립트"""
import pandas as pd

print("=" * 80)
print("1️⃣ 한국어_단발성_대화_데이터셋.xlsx")
print("=" * 80)

df1 = pd.read_excel('raw/한국어_단발성_대화_데이터셋.xlsx')
print(f"\n📊 Shape: {df1.shape}")
print(f"📋 Columns: {df1.columns.tolist()}")
print(f"\n🔍 First 5 rows:")
print(df1.head(5))
print(f"\n📈 Data Info:")
print(df1.info())

# 각 컬럼의 유니크 값 확인
for col in df1.columns:
    unique_count = df1[col].nunique()
    print(f"\n{col}: {unique_count} unique values")
    if df1[col].dtype == 'object' and unique_count < 20:
        print(df1[col].value_counts())

print("\n" + "=" * 80)
print("2️⃣ 한국어_연속적_대화_데이터셋.xlsx")
print("=" * 80)

df2 = pd.read_excel('raw/한국어_연속적_대화_데이터셋.xlsx')
print(f"\n📊 Shape: {df2.shape}")
print(f"📋 Columns: {df2.columns.tolist()}")
print(f"\n🔍 First 5 rows:")
print(df2.head(5))
print(f"\n📈 Data Info:")
print(df2.info())

# 각 컬럼의 유니크 값 확인
for col in df2.columns:
    unique_count = df2[col].nunique()
    print(f"\n{col}: {unique_count} unique values")
    if df2[col].dtype == 'object' and unique_count < 20:
        print(df2[col].value_counts())
