# duct_type_central_hc_system

## 主な変更点（計算結果に関わることのみ記載）
20190423：初バージョン  
20190603：熱源機の送風温度の上限値・下限値を設定  
201907**：VAV無しについて熱源機出口温度の計算を変更
20190805：熱源機処理負荷の計算については概ね完成

## 使い方

### はじめに
- python 3系の最新版(3.7以上推奨）をインストールしてください。
- 少なくとも pandas, numpy, scipy, functools, typing モジュールをインストールする必要があります。
- その他、matplotlib, datetime, os, unittest モジュールを使用しています。
- anaconda 等をインストールするとpython本体とパッケージで上記のモジュールがインストールされるため便利です。

### 使い方

1. 右上のDownloadからすべてのファイルをローカルPCにダウンロードします。（気象データや負荷データ等も含まれるため、300MB以上あります。ご注意ください。）

2. input.json ファイルが入力条件です。
必要に応じて書き換えてください。
値部分は半角小文字、地域区分等の整数は整数で、風量などの小数はきりの良い値であっても小数（例えば1800.0など）で入力してください。
VAVか否か等のboolean型の場合は、半角文字でtrue又はfalseを入力してください。
その他、断熱区分などの文字列は、正確に入力し、ダブルクォーテーションマークで囲ってください。
行の最後には必ず半角でコンマをつけて区切ってください。ただし、最終行はコンマ不要です。

3. main.py を実行すると、result.csv ファイルが作成されれば成功です。

### input.json ファイルの説明
region": 地域の区分です。1から8までの整数値を指定します。

a_mr: 主たる居室の床面積です。単位はm2です。小数を指定します。（例：29.81）

a_or: その他の居室の床面積です。単位はm2です。小数を指定します。（例：51.34）

a_a: 床面積の合計です。単位はm2です。小数を指定します。（例：120.08）

r_env: 床面積の合計に対する外皮の面積の合計です。小数を指定します。（例：2.95556）

insulation: 断熱性能です。下記の中から文字列で指定します。（例："h11"）
- s55: 昭和55年基準レベル
- h4: 平成4年基準レベル
- h11: 平成11年基準レベル
- h11more: 平成11年基準超レベル

solar_gain: 日射熱の取得性能です。下記の中から文字列で指定します。（例："middle"）
- small: 日射熱取得「小」
- middle: 日射熱取得「中」
- large: 日射熱取得「大」

以下の値は、辞書のkey=system_specでくくります。

is_duct_insulated: ダクトが断熱区画内に入っているか否かです。断熱区画内の場合はtrueを、断熱区画外の場合はfalseを指定します。（例：true）

vav_system: VAVシステムを採用するか否かです。採用する場合はtrueを、採用しない場合はfalseを指定します。（例：false）

ventilation_included: セントラル空調が換気システムを含むか否かを指定します。含む場合はtrueを、含まない場合はfalseを指定します。（例：true）

input: 熱源機の性能（熱源機容量）を指定するか否かです。デフォルト値を指定する場合はdefaultを、定格能力試験時の値を入力する場合は"rated"を、定格能力試験時と中間能力試験時の両方の値を入力する場合は"rated_and_middle"を指定します。（例："default"）

cap_rtd_h: 定格暖房能力試験時における能力です。単位はWです。"input"が"default"の場合はこの値は無視されます。

cap_rtd_c: 定格冷房能力試験時における能力です。単位はWです。"input"が"default"の場合はこの値は無視されます。

p_rtd_h: 定格暖房能力試験時における消費電力です。単位はWです。"input"が"default"の場合はこの値は無視されます。

p_rtd_c: 定格冷房能力試験時における消費電力です。単位はWです。"input"が"default"の場合はこの値は無視されます。

v_hs_rtd_h: 定格暖房能力試験時における風量です。単位はm3/hです。"input"が"default"の場合はこの値は無視されます。

v_hs_rrd_c: 定格冷房能力試験時における風量です。単位はm3/hです。"input"が"default"の場合はこの値は無視されます。

p_fan_rtd_h: 定格暖房能力試験時におけるファン消費電力です。単位はWです。"input"が"default"の場合はこの値は無視されます。

p_fan_rtd_c: 定格冷房能力試験時におけるファン消費電力です。単位はWです。"input"が"default"の場合はこの値は無視されます。

q_mid_h: 中間暖房能力試験時における能力です。単位はWです。"input"が"default"又は"rated"の場合はこの値は無視されます。

q_mid_c: 中間冷房能力試験時における能力です。単位はWです。"input"が"default"又は"rated"の場合はこの値は無視されます。

p_mid_h: 中間暖房能力試験時における消費電力です。単位はWです。"input"が"default"又は"rated"の場合はこの値は無視されます。

p_mid_c: 中間冷房能力試験時における消費電力です。単位はWです。"input"が"default"又は"rated"の場合はこの値は無視されます。

v_hs_mid_h: 中間暖房能力試験時における風量です。単位はm3/hです。"input"が"default"又は"rated"の場合はこの値は無視されます。

v_hs_mid_c: 中間冷房能力試験時における風量です。単位はm3/hです。"input"が"default"又は"rated"の場合はこの値は無視されます。

p_fan_mid_h: 中間暖房能力試験時におけるファン消費電力です。単位はWです。"input"が"default"又は"rated"の場合はこの値は無視されます。

p_fan_mid_c: 中間冷房能力試験時におけるファン消費電力です。単位はWです。"input"が"default"又は"rated"の場合はこの値は無視されます。

### input.json ファイルの例

熱源機特性にデフォルト値を用いる場合

{  
    "region": 6,  
    "a_mr": 29.81,  
    "a_or": 51.34,  
    "a_a": 120.08,  
    "r_env": 2.95556,  
    "insulation": "h11",  
    "solar_gain": "middle",  
    "system_spec": {  
      "is_duct_insulated": true,  
      "vav_system": false,  
      "ventilarion_inlculded": true,  
      "input": "default",  
    }  
}

熱源機特性にデフォルト値を用いず、定格能力試験時の値を指定する場合

{  
    "region": 6,  
    "a_mr": 29.81,  
    "a_or": 51.34,  
    "a_a": 120.08,  
    "r_env": 2.95556,  
    "insulation": "h11",  
    "solar_gain": "middle",  
    "system_spec": {  
      "is_duct_insulated": true,  
      "vav_system": false,  
      "ventilarion_inlculded": true,  
      "input": "rated",  
      "cap_rtd_h": 12000.0,  
      "cap_rtd_c": 12000.0,  
      "p_rtd_h": 4000.0,  
      "p_rtd_c": 2400.0,  
      "v_hs_rtd_h": 1800.0,  
      "v_hs_rtd_c": 1800.0,  
      "p_fan_rtd_h": 100.0,  
      "p_fan_rtd_c": 100.0,  
    }  
}  

熱源機特性にデフォルト値を用いず、定格能力試験時の値を指定する場合

{  
    "region": 6,  
    "a_mr": 29.81,  
    "a_or": 51.34,  
    "a_a": 120.08,  
    "r_env": 2.95556,  
    "insulation": "h11",  
    "solar_gain": "middle",  
    "system_spec": {  
      "is_duct_insulated": true,  
      "vav_system": false,  
      "ventilarion_inlculded": true,  
      "input": "rated_and_middle",  
      "cap_rtd_h": 12000.0,  
      "cap_rtd_c": 12000.0,  
      "p_rtd_h": 4000.0,  
      "p_rtd_c": 2400.0,  
      "v_hs_rtd_h": 1800.0,  
      "v_hs_rtd_c": 1800.0,  
      "p_fan_rtd_h": 100.0,  
      "p_fan_rtd_c": 100.0,  
      "q_mid_h": 6000.0,  
      "q_mid_c": 6000.0,  
      "p_mid_h": 1800.0,  
      "p_mid_c": 1000.0,  
      "v_hs_mid_h": 900.0,  
      "v_hs_mid_c": 900.0,  
      "p_fan_mid_h": 50.0,  
      "p_fan_mid_c": 50.0,  
    }  
}  
