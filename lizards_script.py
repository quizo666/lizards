# %%
pip install pandas


# %%
pip install seaborn

# %%
import pandas as pd
import seaborn as sns
import numpy as np

# %%
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('Ящерицы_данные.csv', sep= ';')

# %%
df.head()


# %%
df.info()

# %% [markdown]
# Создайте критерий, позволяющий наилучшим возможным образом отличать ящериц
# вида №5 от всех остальных ящериц и использующий только количество бедренных пор
# справа (FPNr).
# Подсказка: постройте и изучите распределение ящериц по FPNr в зависимости от их
# вида.
# 

# %%
grouped_df = df.groupby('Species_num')['FPNr'].mean()

# %%
grouped_df

# %%
fifth_species= df[df['Species_num']== 5]

# %%
fifth_species

# %%
other_species = df[df['Species_num'] !=5]

# %%
other_species

# %% [markdown]
# Разные метрики:
# 

# %%
fifth_species['FPNr'].mean()

# %%
fifth_species['FPNr'].max()

# %%
other_species['FPNr'].mean()

# %%
other_species['FPNr'].min()

# %%
Q1 = fifth_species['FPNr'].quantile(0.25)
Q3 = fifth_species['FPNr'].quantile(0.75)
IQR = Q3 - Q1

# %%
IQR

# %%
lower_bound = fifth_species['FPNr'].mean() - 1.5*IQR
upper_bound = fifth_species['FPNr'].mean()+ 1.5*IQR
non_outliers_fifth = fifth_species[(fifth_species['FPNr']> lower_bound) & (fifth_species['FPNr']< upper_bound)]

# %%
non_outliers_fifth

# %%
other_Q1 = fifth_species['FPNr'].quantile(0.25)
other_Q3 = fifth_species['FPNr'].quantile(0.75)
other_IQR = other_Q3 - other_Q1

# %%
other_lower_bound = other_species['FPNr'].mean() - 1.5*other_IQR
other_upper_bound = other_species['FPNr'].mean()+ 1.5*other_IQR
non_outliers_other = other_species[(other_species['FPNr']> other_lower_bound) & (other_species['FPNr']< other_upper_bound)]

# %%
non_outliers_other

# %%
non_outliers_other['FPNr'].mean()

# %%
non_outliers_other['FPNr'].min()

# %%
other_species['FPNr'].min()

# %%
non_outliers_fifth['FPNr'].mean()

# %%
non_outliers_fifth['FPNr'].max()

# %%
fifth_species['FPNr'].plot(kind='hist')

# %%
other_species['FPNr'].plot(kind='hist')

# %%
non_outliers_other['FPNr'].plot(kind='hist')

# %%
fifth_species['FPNr'].plot(kind='hist')

# %%
def hypothesis(FPNr):
    if FPNr <=11:
        return 5
    else:
        return 'not 5'

# %%
df['hypothetic_species'] = df['FPNr'].apply(hypothesis)

# %%
hypothetic_df = df[df['hypothetic_species'] == 5]
hypothetic_df

# %%
real_df = df[df['Species_num'] == 5]
real_df

# %%
print(f'Итого, количество предполагаемых особей 5-го вида: {len(hypothetic_df)}')
print(f'Итого, количество реальных особей 5-го вида: {len(real_df)}')

# %% [markdown]
# Создайте критерий, позволяющий наилучшим возможным образом отличать ящериц
# вида №5 от всех остальных ящериц и использующий две переменных из измеряемых
# морфометрических и фолидозных признаков.
# Подсказка: одним из способов нахождения наилучшей пары предсказывающих
# переменных (предикторов) может быть перебор всех возможных пар переменных.
# 

# %%
fifth_species

# %%
fifth_species.info()

# %%
numeric_fifth_species = fifth_species.select_dtypes(include=['int64','float64'])

# %%
numeric_fifth_species

# %%
numeric_other_species = other_species.select_dtypes(include=['int64','float64'])

# %%
fifth_species_metrics = pd.DataFrame({
    'mean': numeric_fifth_species.mean(),
    'median': numeric_fifth_species.median(),
    'std': numeric_fifth_species.std(),
    'min': numeric_fifth_species.min(),
    'max': numeric_fifth_species.max()
})

# %%
fifth_species_metrics

# %%
other_species_metrics = pd.DataFrame({
    'mean': numeric_other_species.mean(),
    'median': numeric_other_species.median(),
    'std': numeric_other_species.std(),
    'min': numeric_other_species.min(),
    'max': numeric_other_species.max()
})

# %%
other_species_metrics


# %%
mean_diff = abs(fifth_species_metrics['mean'] - other_species_metrics['mean'])

# %%
mean_diff.sort_values(ascending=False)

# %%
features = ['MBS', 'FPNr', 'SVL']
pairs = [('MBS', 'FPNr'), ('MBS', 'SVL'), ('FPNr', 'SVL')]

# %%
#FPNr
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
sns.histplot(data=fifth_species, x='FPNr', color='red', label='Species 5', kde=True)
sns.histplot(data=other_species, x='FPNr', color='blue', label='Others', kde=True)
plt.legend()

#MBS
plt.subplot(1, 3, 2)
sns.histplot(data=fifth_species, x='MBS', color='red', label='Species 5', kde=True)
sns.histplot(data=other_species, x='MBS', color='blue', label='Others', kde=True)
plt.legend()

#SVL
plt.subplot(1, 3, 3)
sns.histplot(data=fifth_species, x='SVL', color='red', label='Species 5', kde=True)
sns.histplot(data=other_species, x='SVL', color='blue', label='Others', kde=True)
plt.legend()

plt.show()

# %%
features = ['MBS', 'FPNr', 'SVL']
pairs = [('MBS', 'FPNr'), ('MBS', 'SVL'), ('FPNr', 'SVL')]

best_errors = float('inf')
best_pair = 0
best_h = 0
best_a = 0
best_b = 0
best_c = 0
best_d = 0

for p1, p2 in pairs:
    min_val = int((df[p1] + df[p2]).min())
    max_val = int((df[p1] + df[p2]).max())

    for h in range(min_val, max_val + 1):
        pred_5 = (df[p1] + df[p2]) < h
        tp = len(df[pred_5 & (df['Species_num'] == 5)])  # Правильно вид 5
        fp = len(df[pred_5 & (df['Species_num'] != 5)])  # не 5, но сказали 5
        fn = len(df[~pred_5 & (df['Species_num'] == 5)])  # 5, но сказали не 5
        tn = len(df[~pred_5 & (df['Species_num'] != 5)])  # Правильно не 5
        
        errors = fp + fn
        
        if errors < best_errors:
            best_errors = errors
            best_pair = (p1, p2)
            best_h = h
            best_tp = tp
            best_fp = fp
            best_fn = fn
            best_tn = tn

print("Лучшая пара:", best_pair)
print("Критерий:", best_pair[0], "+", best_pair[1], "<", round(best_h, 1))
print("Таблица ошибок: TP =", best_tp, "FP =", best_fp, "FN =", best_fn, "TN =", best_tn)
print("Всего ошибок:", best_fp + best_fn)
print('Процент ошибок', round((best_fp + best_fn)/len(df)*100,1),'%')

# %% [markdown]
# Создайте критерий, позволяющий наилучшим возможным образом предсказывать пол
# ящериц вне зависимости от их вида по морфометрическим признакам и/или признакам
# фолидоза.
# Подсказка от биологов: предполагается (но не гарантируется!), что пол будет
# взаимосвязан с отношениями некоторых измеряемых длин; но это не исключает
# участия в критерии и других предикторов

# %%
import itertools

# %%
morphometric_cols = ['SVL', 'TRL', 'HL', 'PL', 'ESD', 'HW', 'HH', 'MO', 'FFL', 'HFL']
morph_df = df[morphometric_cols]

# %%

for col1, col2 in itertools.combinations(morphometric_cols, 2):
    ratio_name = col1 + '_to_' + col2
    morph_df[ratio_name] = df[col1] / df[col2]


male_mean = morph_df[df['Sex_num'] == 1].mean()
female_mean = morph_df[df['Sex_num'] == 2].mean()

mean_diff = abs(male_mean - female_mean)
top_features = mean_diff.sort_values(ascending=False).index[:6].tolist()
pairs = list(itertools.combinations(top_features, 2))

best_errors = float('inf')
best_pair = 0
best_h = 0
best_tp = 0
best_fp = 0
best_fn = 0
best_tn = 0

for p1, p2 in pairs:
    values = morph_df[p1] / morph_df[p2]
    thresholds = [values.quantile(q) for q in [0.25, 0.5, 0.75]]
    for h in thresholds:
        is_male = values < h
        tp = len(df[is_male & (df['Sex_num'] == 1)])
        fp = len(df[is_male & (df['Sex_num'] == 2)])
        fn = len(df[~is_male & (df['Sex_num'] == 1)])
        tn = len(df[~is_male & (df['Sex_num'] == 2)])
        errors = fp + fn
        if errors < best_errors:
            best_errors = errors
            best_pair = (p1, p2)
            best_h = h
            best_tp = tp
            best_fp = fp
            best_fn = fn
            best_tn = tn

print("Лучшая пара:", best_pair)
print("Критерий:", best_pair[0], "/", best_pair[1], "<", round(best_h, 1))
print("Таблица ошибок: TP =", best_tp, "FP =", best_fp, "FN =", best_fn, "TN =", best_tn)
print("Всего ошибок:", best_fp + best_fn)
print("Процент ошибок:", round((best_fp + best_fn) / len(df) * 100, 1),"%")



# %% [markdown]
# Задание 4: Не все рассматриваемые виды ящериц встречаются в одних и тех же местах обитания.
# Поэтому на практике чаще всего встречаются задачи различения видов из
# определённых подгрупп, обитающих совместно. Создайте набор критериев,
# позволяющих наилучшим возможным образом отличать друг от друга все виды внутри
# следующих групп:
# a) виды №6 и №7,
# b) виды №1 и №2,
# c) виды №3, №4 и №5.
# 

# %%
features = ['SVL', 'TRL', 'HL', 'PL', 'ESD', 'HW', 'HH', 'MO', 'FFL', 'HFL', 'FPNr', 'MBS']

def find_criterion_two_species(df, species1, species2, group_name):
    df_group = df[df['Species_num'].isin([species1, species2])].copy()
    mean1 = df_group[df_group['Species_num'] == species1][features].mean()
    mean2 = df_group[df_group['Species_num'] == species2][features].mean()
    mean_diff = abs(mean1 - mean2)
    best_feature = mean_diff.idxmax()
    values = df_group[best_feature]
    thresholds = [values.quantile(q) for q in [0.25, 0.5, 0.75]]
    best_errors = float('inf')
    best_h = 0
    best_tp = 0
    best_fp = 0
    best_fn = 0
    best_tn = 0
    for h in thresholds:
        is_species1 = values < h
        tp = len(df_group[is_species1 & (df_group['Species_num'] == species1)])
        fp = len(df_group[is_species1 & (df_group['Species_num'] == species2)])
        fn = len(df_group[~is_species1 & (df_group['Species_num'] == species1)])
        tn = len(df_group[~is_species1 & (df_group['Species_num'] == species2)])
        errors = fp + fn
        if errors < best_errors:
            best_errors = errors
            best_h = h
            best_tp = tp
            best_fp = fp
            best_fn = fn
            best_tn = tn
    print(f"\nГруппа {group_name}: Виды {species1} и {species2}")
    print(f"Признак: {best_feature}")
    print(f"Критерий: {best_feature} < {round(best_h, 1)} (меньше - вид {species1}, иначе - вид {species2})")
    print(f"Таблица ошибок: TP = {best_tp}, FP = {best_fp}, FN = {best_fn}, TN = {best_tn}")
    print(f"Всего ошибок: {best_fp + best_fn}")
    print(f"Процент ошибок: {round((best_fp + best_fn) / len(df_group) * 100, 1)} %")

def find_criterion_three_species(df):
    df_group = df[df['Species_num'].isin([3, 4, 5])].copy()
    values_fpnr = df_group['FPNr']
    h_fpnr = 11
    is_species5 = values_fpnr <= h_fpnr
    tp5 = len(df_group[is_species5 & (df_group['Species_num'] == 5)])
    fp5 = len(df_group[is_species5 & (df_group['Species_num'] != 5)])
    fn5 = len(df_group[~is_species5 & (df_group['Species_num'] == 5)])
    tn5 = len(df_group[~is_species5 & (df_group['Species_num'] != 5)])
    print("\nГруппа c: Виды 3, 4, 5")
    print("Критерий 1: Вид 5")
    print(f"Признак: FPNr")
    print(f"Критерий: FPNr <= {h_fpnr} (меньше или равно - вид 5)")
    print(f"Таблица ошибок: TP = {tp5}, FP = {fp5}, FN = {fn5}, TN = {tn5}")
    print(f"Всего ошибок: {fp5 + fn5}")
    print(f"Процент ошибок: {round((fp5 + fn5) / len(df_group) * 100, 1)} %")
    df_34 = df_group[~is_species5].copy()
    if len(df_34) == 0:
        print("Нет ящериц для разделения видов 3 и 4")
        return
    mean3 = df_34[df_34['Species_num'] == 3][features].mean()
    mean4 = df_34[df_34['Species_num'] == 4][features].mean()
    mean_diff = abs(mean3 - mean4)
    best_feature = mean_diff.idxmax()
    values = df_34[best_feature]
    thresholds = [values.quantile(q) for q in [0.25, 0.5, 0.75]]
    best_errors = float('inf')
    best_h = 0
    best_tp = 0
    best_fp = 0
    best_fn = 0
    best_tn = 0
    for h in thresholds:
        is_species3 = values < h
        tp = len(df_34[is_species3 & (df_34['Species_num'] == 3)])
        fp = len(df_34[is_species3 & (df_34['Species_num'] == 4)])
        fn = len(df_34[~is_species3 & (df_34['Species_num'] == 3)])
        tn = len(df_34[~is_species3 & (df_34['Species_num'] == 4)])
        errors = fp + fn
        if errors < best_errors:
            best_errors = errors
            best_h = h
            best_tp = tp
            best_fp = fp
            best_fn = fn
            best_tn = tn
    print("\nКритерий 2: Виды 3 и 4 (после отделения вида 5)")
    print(f"Признак: {best_feature}")
    print(f"Критерий: {best_feature} < {round(best_h, 1)} (меньше - вид 3, иначе - вид 4)")
    print(f"Таблица ошибок: TP = {best_tp}, FP = {best_fp}, FN = {best_fn}, TN = {best_tn}")
    print(f"Всего ошибок: {best_fp + best_fn}")
    print(f"Процент ошибок: {round((best_fp + best_fn) / len(df_34) * 100, 1)} %")

find_criterion_two_species(df, 6, 7, "a")
find_criterion_two_species(df, 1, 2, "b")
find_criterion_three_species(df)

# %% [markdown]
# Задание 5:Создайте критерий или набор критериев, позволяющий наилучшим возможным
# образом предсказывать вид или вид и пол ящериц во всей их совокупности (это может
# понадобиться биологам, если они не знают место отлова ящерицы). Вполне возможно,
# что некоторые пары или группы видов не будут разделяться на основе имеющихся
# данных. Приведите наилучший полученный результат, который может в наибольшей
# степени помочь биологам.
# 

# %%
features = ['SVL', 'TRL', 'HL', 'PL', 'ESD', 'HW', 'HH', 'MO', 'FFL', 'HFL', 'FPNr', 'MBS']
morph_df = df[features].copy()
for col1, col2 in itertools.combinations(['SVL', 'HL', 'FFL'], 2):
    ratio_name = col1 + '_to_' + col2
    morph_df[ratio_name] = df[col1] / df[col2]
features.extend(['SVL_to_HL', 'SVL_to_FFL', 'HL_to_FFL'])

def classify_species_and_sex(df, morph_df):
    df_remaining = df.copy()
    morph_df_remaining = morph_df.loc[df_remaining.index].copy()
    predictions = pd.Series('unknown', index=df.index)
    sex_predictions = pd.Series('unknown', index=df.index)
    
    # 5 вид
    values_fpnr = df_remaining['FPNr']
    h_fpnr = 11
    is_species5 = values_fpnr <= h_fpnr
    predictions[df_remaining[is_species5].index] = 'Species 5'
    
    tp5 = len(df_remaining[is_species5 & (df_remaining['Species_num'] == 5)])
    fp5 = len(df_remaining[is_species5 & (df_remaining['Species_num'] != 5)])
    fn5 = len(df_remaining[~is_species5 & (df_remaining['Species_num'] == 5)])
    tn5 = len(df_remaining[~is_species5 & (df_remaining['Species_num'] != 5)])
    
    print("\nШаг 1: Вид 5")
    print(f"Критерий: FPNr <= {h_fpnr}")
    print(f"Таблица ошибок: TP = {tp5}, FP = {fp5}, FN = {fn5}, TN = {tn5}")
    print(f"Всего ошибок: {fp5 + fn5}")
    print(f"Процент ошибок: {round((fp5 + fn5) / len(df_remaining) * 100, 1)} %")
    
    # минус 5
    df_remaining = df_remaining[~is_species5].copy()
    morph_df_remaining = morph_df_remaining.loc[df_remaining.index].copy()
    
    # вилы 6 и 7
    df_67 = df_remaining[df_remaining['Species_num'].isin([6, 7])].copy()
    morph_df_67 = morph_df_remaining.loc[df_67.index].copy()
    
    if len(df_67) > 0:
        mean6 = morph_df_67[df_67['Species_num'] == 6][features].mean()
        mean7 = morph_df_67[df_67['Species_num'] == 7][features].mean()
        mean_diff = abs(mean6 - mean7)
        top_features = mean_diff.sort_values(ascending=False).index[:2].tolist()
        p1, p2 = top_features
        
        values1 = morph_df_67[p1]
        values2 = morph_df_67[p2]
        thresholds1 = [values1.quantile(q) for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
        thresholds2 = [values2.quantile(q) for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
        
        best_errors = float('inf')
        best_a = 0
        best_b = 0
        best_tp = 0
        best_fp = 0
        best_fn = 0
        best_tn = 0
        best_op = None
        
        for a in thresholds1:
            for b in thresholds2:
                for op in ['multiply', 'and']:
                    if op == 'multiply':
                        values = values1 * values2
                        is_species6 = values < b
                    else:
                        is_species6 = (values1 < a) & (values2 < b)
                    tp = len(df_67[is_species6 & (df_67['Species_num'] == 6)])
                    fp = len(df_67[is_species6 & (df_67['Species_num'] == 7)])
                    fn = len(df_67[~is_species6 & (df_67['Species_num'] == 6)])
                    tn = len(df_67[~is_species6 & (df_67['Species_num'] == 7)])
                    errors = fp + fn
                    if errors < best_errors:
                        best_errors = errors
                        best_a = a
                        best_b = b
                        best_tp = tp
                        best_fp = fp
                        best_fn = fn
                        best_tn = tn
                        best_op = op
                        best_p1 = p1
                        best_p2 = p2
        
        predictions[df_67[is_species6].index] = 'Species 6'
        predictions[df_67[~is_species6].index] = 'Species 7'
        
        print("\nШаг 2: Виды 6 и 7")
        print(f"Признаки: {best_p1}, {best_p2}")
        if best_op == 'multiply':
            print(f"Критерий: {best_p1} * {best_p2} < {round(best_b, 1)} (меньше - вид 6)")
        else:
            print(f"Критерий: {best_p1} < {round(best_a, 1)} и {best_p2} < {round(best_b, 1)} (меньше - вид 6)")
        print(f"Таблица ошибок: TP = {best_tp}, FP = {best_fp}, FN = {best_fn}, TN = {best_tn}")
        print(f"Всего ошибок: {best_fp + best_fn}")
        print(f"Процент ошибок: {round((best_fp + best_fn) / len(df_67) * 100, 1)} %")
    
    # минус 6 и 7
    df_remaining = df_remaining[~df_remaining['Species_num'].isin([6, 7])].copy()
    morph_df_remaining = morph_df_remaining.loc[df_remaining.index].copy()
    
    # виды 3 и 4
    df_34 = df_remaining[df_remaining['Species_num'].isin([3, 4])].copy()
    morph_df_34 = morph_df_remaining.loc[df_34.index].copy()
    
    if len(df_34) > 0:
        mean3 = morph_df_34[df_34['Species_num'] == 3][features].mean()
        mean4 = morph_df_34[df_34['Species_num'] == 4][features].mean()
        mean_diff = abs(mean3 - mean4)
        top_features = mean_diff.sort_values(ascending=False).index[:2].tolist()
        p1, p2 = top_features
        
        values1 = morph_df_34[p1]
        values2 = morph_df_34[p2]
        thresholds1 = [values1.quantile(q) for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
        thresholds2 = [values2.quantile(q) for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
        
        best_errors = float('inf')
        best_a = 0
        best_b = 0
        best_tp = 0
        best_fp = 0
        best_fn = 0
        best_tn = 0
        best_op = None
        
        for a in thresholds1:
            for b in thresholds2:
                for op in ['multiply', 'and']:
                    if op == 'multiply':
                        values = values1 * values2
                        is_species3 = values < b
                    else:
                        is_species3 = (values1 < a) & (values2 < b)
                    tp = len(df_34[is_species3 & (df_34['Species_num'] == 3)])
                    fp = len(df_34[is_species3 & (df_34['Species_num'] == 4)])
                    fn = len(df_34[~is_species3 & (df_34['Species_num'] == 3)])
                    tn = len(df_34[~is_species3 & (df_34['Species_num'] == 4)])
                    errors = fp + fn
                    if errors < best_errors:
                        best_errors = errors
                        best_a = a
                        best_b = b
                        best_tp = tp
                        best_fp = fp
                        best_fn = fn
                        best_tn = tn
                        best_op = op
                        best_p1 = p1
                        best_p2 = p2
        
        predictions[df_34[is_species3].index] = 'Species 3'
        predictions[df_34[~is_species3].index] = 'Species 4'
        
        print("\nШаг 3: Виды 3 и 4")
        print(f"Признаки: {best_p1}, {best_p2}")
        if best_op == 'multiply':
            print(f"Критерий: {best_p1} * {best_p2} < {round(best_b, 1)} (меньше - вид 3)")
        else:
            print(f"Критерий: {best_p1} < {round(best_a, 1)} и {best_p2} < {round(best_b, 1)} (меньше - вид 3)")
        print(f"Таблица ошибок: TP = {best_tp}, FP = {best_fp}, FN = {best_fn}, TN = {best_tn}")
        print(f"Всего ошибок: {best_fp + best_fn}")
        print(f"Процент ошибок: {round((best_fp + best_fn) / len(df_34) * 100, 1)} %")
    
    # виды 1 и 2
    df_12 = df_remaining[df_remaining['Species_num'].isin([1, 2])].copy()
    if len(df_12) > 0:
        predictions[df_12.index] = 'Species 1 or 2'
        print("\nШаг 4: Виды 1 и 2")
        print("Критерий: Виды 1 и 2 неразделимы, классифицируем как 'Species 1 or 2'")
        print(f"Количество ящериц: {len(df_12)}")
    
    # пол вида номер 5
    df_species5 = df[predictions == 'Species 5'].copy()
    morph_df_species5 = morph_df.loc[df_species5.index].copy()
    
    if len(df_species5) > 0:
        mean_male = morph_df_species5[df_species5['Sex_num'] == 1][features].mean()
        mean_female = morph_df_species5[df_species5['Sex_num'] == 2][features].mean()
        mean_diff = abs(mean_male - mean_female)
        top_features = mean_diff.sort_values(ascending=False).index[:2].tolist()
        p1, p2 = top_features
        
        values1 = morph_df_species5[p1]
        values2 = morph_df_species5[p2]
        thresholds1 = [values1.quantile(q) for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
        thresholds2 = [values2.quantile(q) for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
        
        best_errors = float('inf')
        best_a = 0
        best_b = 0
        best_tp = 0
        best_fp = 0
        best_fn = 0
        best_tn = 0
        best_op = None
        
        for a in thresholds1:
            for b in thresholds2:
                for op in ['multiply', 'and']:
                    if op == 'multiply':
                        values = values1 * values2
                        is_male = values < b
                    else:
                        is_male = (values1 < a) & (values2 < b)
                    tp = len(df_species5[is_male & (df_species5['Sex_num'] == 1)])
                    fp = len(df_species5[is_male & (df_species5['Sex_num'] == 2)])
                    fn = len(df_species5[~is_male & (df_species5['Sex_num'] == 1)])
                    tn = len(df_species5[~is_male & (df_species5['Sex_num'] == 2)])
                    errors = fp + fn
                    if errors < best_errors:
                        best_errors = errors
                        best_a = a
                        best_b = b
                        best_tp = tp
                        best_fp = fp
                        best_fn = fn
                        best_tn = tn
                        best_op = op
                        best_p1 = p1
                        best_p2 = p2
        
        sex_predictions[df_species5[is_male].index] = 'Male'
        sex_predictions[df_species5[~is_male].index] = 'Female'
        
        print("\nШаг 5: Пол для вида 5")
        print(f"Признаки: {best_p1}, {best_p2}")
        if best_op == 'multiply':
            print(f"Критерий: {best_p1} * {best_p2} < {round(best_b, 1)} (меньше - самец)")
        else:
            print(f"Критерий: {best_p1} < {round(best_a, 1)} и {best_p2} < {round(best_b, 1)} (меньше - самец)")
        print(f"Таблица ошибок: TP = {best_tp}, FP = {best_fp}, FN = {best_fn}, TN = {best_tn}")
        print(f"Всего ошибок: {best_fp + best_fn}")
        print(f"Процент ошибок: {round((best_fp + best_fn) / len(df_species5) * 100, 1)} %")
    
    # ошибки
    species_errors = sum(predictions != df['Species_num'].map({1: 'Species 1 or 2', 2: 'Species 1 or 2', 3: 'Species 3', 4: 'Species 4', 5: 'Species 5', 6: 'Species 6', 7: 'Species 7'}))
    print("\nИтоговые результаты")
    print(f"Общее количество ошибок классификации видов: {species_errors}")
    print(f"Процент ошибок для видов: {round(species_errors / len(df) * 100, 1)} %")

classify_species_and_sex(df, morph_df)

# %%



