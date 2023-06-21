from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

@app.route('/')
def display_dataframe():
    # データフレームを作成
    data = {
        'Name': ['John', 'Alice', 'Bob', 'Eve'],
        'Age': [28, 24, 22, 24],
        'City': ['New York', 'Los Angeles', 'San Francisco', 'Seattle']
    }
    df = pd.DataFrame(data)

    # データフレームをHTMLテーブルに変換
    html_table = df.to_html(classes='table table-striped', index=False, header=False, table_id='dataframe')


    return render_template('index.html', table=html_table, df_columns=df.columns)

if __name__ == '__main__':
    app.run(debug=True)
