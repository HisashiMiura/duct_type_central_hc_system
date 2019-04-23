# duct_type_central_hc_system

## 使い方

### はじめに
- python 3系の最新版(3.7以上推奨）をインストールしてください。
- 少なくとも pandas, numpy, functools, typing モジュールをインストールする必要があります。
- その他、matplotlib, datetime, os, unittest モジュールを使用しています。
- anaconda 等をインストールするとpython本体とパッケージで上記のモジュールがインストールされるため便利です。

### 使い方

1. 右上のDownloadからすべてのファイルをローカルPCにダウンロードします。

2. input.json ファイルが入力条件です。
必要に応じて書き換えてください。
値部分は半角小文字、地域区分等の整数は整数で、風量などの小数はきりの良い値であっても小数（例えば1800.0など）で入力してください。
VAVか否か等のboolean型の場合は、半角小文字でtrue又はfalseを入力してください。
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

default_heat_source_spec: 熱源機の性能（熱源機容量）を指定するか否かです。デフォルト値を指定する場合はtrueを、入力する場合はfalseを指定します。（例：true）<br>
なお、ここでtrueを指定した場合は、cap_rtd_h及びcap_rtd_cに値が指定されていたとしてもその値は参照されません。

supply_air_rtd_h: 暖房時定格風量です。単位はm3/hです。小数を指定します。（例：1800.0）

supply_air_rtd_c: 冷房時定格風量です。単位はm3/hです。小数を指定します。（例：1800.0）

is_duct_insulated: ダクトが断熱区画内に入っているか否かです。断熱区画内の場合はtrueを、断熱区画外の場合はfalseを指定します。（例：true）

vav_system: VAVシステムを採用するか否かです。採用する場合はtrueを、採用しない場合はfalseを指定します。（例：false）

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
    "default_heat_source_spec": true,
    "supply_air_rtd_h": 1800.0,
    "supply_air_rtd_c": 1800.0,
    "is_duct_insulated": true,
    "vav_system": false,
    "cap_rtd_h": 12000.0,
    "cap_rtd_c": 12000.0
}

熱源機特性にデフォルト値を用いず、個別の値を指定する場合

{
    "region": 6,
    "a_mr": 29.81,
    "a_or": 51.34,
    "a_a": 120.08,
    "r_env": 2.95556,
    "insulation": "h11",
    "solar_gain": "middle",
    "default_heat_source_spec": false,
    "supply_air_rtd_h": 1800.0,
    "supply_air_rtd_c": 1800.0,
    "is_duct_insulated": true,
    "vav_system": false,
    "cap_rtd_h": 12000.0,
    "cap_rtd_c": 12000.0
}
