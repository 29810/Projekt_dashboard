import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_option_menu import option_menu
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import seaborn as sns


pd.set_option('display.max_columns', None)
df = pd.read_csv("/Users/joanna/Documents/PJA/messy_data.csv")

def cleanData():
    df = pd.read_csv("/Users/joanna/Documents/PJA/messy_data.csv")
    df.columns = df.columns.str.lstrip()

    # Zamiana pozostałych spacji nazw na _
    df.columns = df.columns.str.replace(' ', '_')
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    # Zmiana typu object na float64
    df['x_dimension'] = df['x_dimension'].astype("float")
    df['y_dimension'] = df['y_dimension'].astype("float64")
    df['z_dimension'] = df['z_dimension'].astype("float64")
    df['depth'] = df['depth'].astype("float64")

    # Zmiana typu object na float64 (ale później będzie na int64)
    df['table'] = df['table'].astype("float64")
    df['price'] = df['price'].astype("float64")

    # Zastąpienie wartości NaN na średnią
    df['x_dimension'] = df['x_dimension'].fillna(df['x_dimension'].mean())
    df['y_dimension'] = df['y_dimension'].fillna(df['y_dimension'].mean())
    df['z_dimension'] = df['z_dimension'].fillna(df['z_dimension'].mean())
    df['depth'] = df['depth'].fillna(df['depth'].mean())
    df['price'] = df['price'].fillna(df['price'].mean())
    df['table'] = df['table'].fillna(df['table'].mean())
    df['carat'] = df['carat'].fillna(df['carat'].mean())

    # Zmiana typu object na int64

    df['table'] = df['table'].astype("int64")
    df['price'] = df['price'].astype("int64")

    # Uporządkowanie danych kategorycznych
    df["clarity"].replace("I1", "i1", regex=True, inplace=True)
    df["clarity"].replace(["Si2", "SI2"], "si2", regex=True, inplace=True)
    df["clarity"].replace(["VVS1", "Vvs1"], "vvs1", regex=True, inplace=True)
    df["clarity"].replace("IF", "if", regex=True, inplace=True)
    df["clarity"].replace("VVS2", "vvs2", regex=True, inplace=True)
    df["clarity"].replace("Si1", "si1", regex=True, inplace=True)

    df["color"].replace("g", "G", regex=True, inplace=True)
    df["color"].replace("h", "H", regex=True, inplace=True)
    df["color"].replace("f", "F", regex=True, inplace=True)
    df["color"].replace("j", "J", regex=True, inplace=True)
    df["color"].replace("e", "E", regex=True, inplace=True)
    df["color"].replace("d", "D", regex=True, inplace=True)
    df["color"].replace(["colorlEss", "ColorlEss"], "colorless", regex=True, inplace=True)

    df["cut"].replace("Premium", "premium", regex=True, inplace=True)
    df["cut"].replace("Good", "good", regex=True, inplace=True)
    df["cut"].replace("Fair", "fair", regex=True, inplace=True)
    df["cut"].replace("Ideal", "ideal", regex=True, inplace=True)
    df["cut"].replace(["Very good", "very good"], "very_good", regex=True, inplace=True)

    #Zmiana typu object na category
    df['clarity'] = df['clarity'].astype("category")
    df['color'] = df['color'].astype("category")
    df['cut'] = df['cut'].astype("category")

    # Pozbycie się duplikatów
    duplicates = df.duplicated(subset=["carat", "table", "price"], keep=False)

    # Na histogramie atrybutu cena możemy zaobserwować duże wartości odstające, których należy się pozbyć
    q_low = df["price"].quantile(0.01)
    q_hi  = df["price"].quantile(0.96)
    df_filtered = df[(df["price"] < q_hi) & (df["price"] > q_low)]
    df = df_filtered
    return df

st.title("PAD messy data demo")

selected = option_menu(
        menu_title=None,
        options = ['Próbka danych', 'Wizualizacja danych', 'Wizualizacja modelu'],
        menu_icon = 'cast',
        default_index=0,
        orientation='horizontal'
    )

if selected == 'Próbka danych':

    st.subheader('Dane przed fazą czyszczenia i analizy:')
    st.dataframe(df.head(10))

    df = cleanData()
    st.subheader('Dane po fazie czyszczenia i analizy:')
    st.dataframe(df.head(10))

elif selected == 'Wizualizacja danych':
    st.subheader('Wizualizacja rozkładu zmiennych')
    df = cleanData()

    st.set_option('deprecation.showPyplotGlobalUse', False)

    columns = df.columns
    column = st.selectbox("Wybierz atrybut:",columns, key=0)
    plt.hist(df[column])
    plt.title(f"Histogram {column}")
    st.pyplot()

    columnsBox = ['carat', 'x_dimension', 'y_dimension', 'z_dimension', 'depth', 'table', 'price']
    columnBox = st.selectbox("Wybierz atrybut:",columnsBox, key=10)
    plt.boxplot(df[columnBox])
    plt.title(f"Boxplot {columnBox}")
    st.pyplot()

    st.subheader('Zalenżość ceny od inncyh zmienncyh')
    columns2 = ['carat', 'x_dimension', 'y_dimension', 'depth', 'table', 'clarity', 'color']
    column2 = st.selectbox("Wybierz atrybut:",columns2, key=1)
    plt.scatter(df[column2], df['price'])
    plt.title(f"Scatter plot {column2}")
    st.pyplot()

    st.subheader('Mapa ciepła')
    dummydf = pd.get_dummies(df, columns = ['clarity'])
    dummydf = pd.get_dummies(dummydf, columns = ['color'])
    dummydf = pd.get_dummies(dummydf, columns = ['cut'])
    figHeat = px.imshow(dummydf.corr(), color_continuous_scale='Agsunset',text_auto=True)
    figHeat.update_layout(height=1000, width=1000)
    st.plotly_chart(figHeat)


    st.subheader('Liczebność kategorii')
    columns3 = ['clarity', 'color', 'cut']
    clarityAmount = len(df['clarity'].unique())
    colorAmount = len(df['color'].unique())
    cutAmount = len(df['cut'].unique())

    plt.bar(columns3, [clarityAmount, colorAmount, cutAmount])
    st.pyplot()

    checks = st.columns(4)
    with checks[0]:
        if st.checkbox('Clarity'):
             dfClarity = pd.DataFrame(df['clarity'].unique(), columns=['Clarity'])
             st.dataframe(dfClarity)
    with checks[1]:
        if st.checkbox('Color'):
            dfColor = pd.DataFrame(df['color'].unique(), columns=['Color'])
            st.dataframe(dfColor)
    with checks[2]:
        if st.checkbox('Cut'):
            dfCut = pd.DataFrame(df['cut'].unique(), columns=['Cut'])
            st.dataframe(dfCut)

elif selected == 'Wizualizacja modelu':
    options = ['carat', 'x_dimension', 'y_dimension', 'depth', 'table', 'clarity', 'color']
    option = st.selectbox("Wybierz atrybut:",options, key=5)

    if option == 'clarity':
        df = cleanData()
        # df['clarity'] = df['clarity'].astype(str)
        # df['clarity'].replace(['if', 'vvs2', 'si2', 'i1', 'si1', 'vvs1'],[0, 1, 2, 3, 4, 5], inplace=True, regex=True)
        model2 = smf.ols("price ~ clarity", data=df).fit()
        df['fitted'] = model2.fittedvalues
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['clarity'], y=df['price'], mode='markers'))
        fig.add_trace(go.Scatter(x=df['clarity'], y=df['fitted']))
        st.plotly_chart(fig)
        st.subheader('Summary modelu:')
        st.write(model2.summary())

    elif  option == 'color':
        df = cleanData()
        model2 = smf.ols(f"price ~ C(color)", data=df).fit()
        df['fitted'] = model2.fittedvalues
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['color'], y=df['price'], mode='markers'))
        fig.add_trace(go.Scatter(x=df['color'], y=df['fitted']))
        st.plotly_chart(fig)
        st.subheader('Summary modelu:')
        st.write(model2.summary())

    else:
        df = cleanData()
        model2 = smf.ols(f"price ~ {option}", data=df).fit()
        df['fitted'] = model2.fittedvalues
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[option], y=df['price'], mode='markers'))
        fig.add_trace(go.Scatter(x=df[option], y=df['fitted']))
        st.plotly_chart(fig)
        st.subheader('Summary modelu:')
        st.write(model2.summary())

    df = cleanData()
    model = smf.ols("price ~ carat + x_dimension + y_dimension + depth + table + C(clarity) + C(color)", data=df).fit()
    st.subheader('Summary modelu z wieloma zmiennymi:')
    st.write(model.summary())
