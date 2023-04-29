# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)

import streamlit as st
import pickle
from datetime import datetime
startTime = datetime.now()

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

filename = "model.sv"
model = pickle.load(open(filename,'rb'))

def main():
	st.set_page_config(page_title="Zdrowie App")
	overview = st.container()
	left, right = st.columns(2)
	prediction = st.container()

	st.image("https://welcome.uw.edu.pl/content/uploads/2017/07/apple-blue-background-close-up-1353366.jpg?1682640000030")

	with overview:
		st.title("Zdrowie App")

	with left:
		symptoms_slider = st.slider("Objawy", value=1, min_value=1, max_value=5)
		age_slider = st.slider("Wiek", value=1, min_value=11, max_value=77)
		other_illnesses_slider = st.slider("Choroby współwystępujące", value=0, min_value=0, max_value=5)

	with right:
		height_slider = st.slider("Wzrost", value=159, min_value=159, max_value=200)
		drugs_slider = st.slider("Leki", value=1, min_value=1, max_value=4)

	data = [[symptoms_slider, age_slider,  other_illnesses_slider, height_slider, drugs_slider]]
	survival = model.predict(data)
	s_confidence = model.predict_proba(data)

	with prediction:
		st.subheader("Czy taka osoba jest zdrowa?")
		st.subheader(("Tak" if survival[0] == 1 else "Nie"))
		st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))

if __name__ == "__main__":
    main()
