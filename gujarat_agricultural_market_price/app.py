import streamlit as st
import sys
import pandas as pd
import joblib


def main():
    st.title("Agriculture Commodity Arrival Prediction")

    district_list = ['Amreli', 'Morbi', 'Sabarkantha', 'Ahmedabad', 'Kachchh', 'Bharuch', 'Mehsana', 'Banaskanth',
                     'Bhavnagar', 'Navsari', 'Chhota Udaipur', 'Botad', 'Surendranagar', 'Dahod', 'Gandhinagar',
                     'Jamnagar', 'Rajkot', 'Kheda', 'Vadodara(Baroda)', 'Gir Somnath', 'Panchmahals', 'Junagarh',
                     'Surat', 'Patan', 'Anand', 'Valsad', 'Narmada', 'Devbhumi Dwarka', 'The Dangs', 'Porbandar', 'Dang']

    market_list = ['APMC Bagasara', 'APMC HALVAD', 'APMCKHEDBRAHMA', 'Ahmedabad APMC',
                   'Ahmedabad(Chimanbhai Patal Market Vasana) APMC', 'Ahmedabad(Fruit Market, Naroda) APMC',
                   'Ahmedabad(Rajnagar sub yard) APMC', 'Anjar APMC', 'Ankleshwar APMC', 'Babra APMC', 'Becharaji APMC',
                   'Bhabhar APMC', 'Bhavnagar APMC', 'Bhiloda APMC', 'Bilimora APMC', 'Bodeliu APMC', 'Botad APMC',
                   'Botad(Haddad) APMC', 'Chikhali APMC', 'Chotila APMC', 'Dahod APMC', 'Davgadbaria(Piplod) APMC',
                   'Deesa APMC', 'Deesa(Bhildi) APMC', 'Deesa(Deesa Veg Yard) APMC', 'Dehgam APMC', 'Devgadhbaria APMC',
                   'Dhanera APMC', 'Dhansura APMC', 'Dhari APMC', 'Dholka APMC', 'Dhrol APMC', 'Gondal APMC',
                   'Hadad     APMC', 'Himatnagar APMC', 'Idar APMC', 'Jamkandorna APMC', 'Jamnagar APMC',
                   'Jetpur(Dist.Rajkot) APMC', 'Jetpur-Pavi APMC', 'Jhagadiya APMC', 'Kadi APMC', 'Kalediya APMC',
                   'Kalol APMC', 'Kapadvanj APMC', 'Karjan APMC', 'Kodinar APMC', 'Limdi APMC', 'Lunavada APMC',
                   'Mangrol APMC', 'Mansa APMC', 'Matar APMC', 'Mehsana APMC', 'Mehsana(Jornang) APMC',
                   'Mehsana(Mehsana Veg) APMC', 'Modasa APMC', 'Morbi APMC', 'Nasvadi APMC', 'Nizar APMC', 'Padra APMC',
                   'Palitana APMC', 'Patan APMC', 'Prantij APMC', 'Rajkot APMC', 'Rajula APMC', 'Rapar APMC',
                   'S.Mandvi APMC', 'Sanad APMC', 'Savarkundla APMC', 'Siddhpur APMC', 'Surat APMC', 'Talod APMC',
                   'Thalkala APMC', 'Thara APMC', 'Tharad APMC', 'Thasara APMC',
                   'The Agricultural Produce Market Committee-Amreli', 'Una APMC', 'Unava APMC', 'Vadhvan APMC',
                   'Vadodara APMC', 'Vadodara(Navapura) APMC', 'Vadodara(Sayajipura) APMC', 'Vankaner APMC',
                   'Vansda APMC', 'Vijapur APMC', 'Visnagar APMC', 'Vyara(Paati) APMC', 'Vyra APMC',
                   'Anand(Veg,Yard,Anand) APMC', 'Bardoli APMC', 'Bhuj APMC', 'Dahod(Veg. Market) APMC', 'Dhandhuka APMC',
                   'Dhoraji APMC', 'Junagadh APMC', 'K.Mandvi APMC', 'Kalawad APMC', 'Mansa(Manas Veg Yard) APMC',
                   'Talalagir APMC', 'Umreth APMC', 'Vadgam APMC', 'Bharuch APMC', 'Jasdan APMC', 'Taleja APMC',
                   'Upleta APMC', 'Visavadar APMC', 'Mahuva APMC', 'Mehmadabad APMC', 'Palanpur APMC', 'Chachack  APMC',
                   'Gogamba APMC', 'Harij APMC', 'Kosamba APMC', 'Amod APMC', 'Mahuva(Anaval) APMC', 'Jam Jodhpur APMC',
                   'Kapadanj(Moti Jaher) APMC', 'Valod(Buhari) APMC', 'Chikli(Khorgam) APMC', 'Valsad APMC', 'Uchhal APMC',
                   'Lalpur APMC', 'Ahmedabad(Manekchowk) APMC', 'Savli APMC', 'Rajpipla APMC', 'Jam Khambalia APMC',
                   'Vagodiya APMC', 'Borsad APMC', 'Godhra APMC', 'Kheda APMC', 'Meghraj APMC', 'Modasar APMC',
                   'Prantij(Salala) APMC', 'Modasa(Tintoi) APMC', 'Thasara(Dokar) APMC', 'Patan(Veg,Yard Patan) APMC',
                   'Mahuva(Station Road) APMC', 'Matar(Limbasi) APMC', 'Savli(Desar) APMC', 'Bhesan APMC', 'Malpur APMC',
                   'Malia Hatina APMC', 'Amirgadh APMC', 'Dholka(Market Yard Veg) APMC', 'Vadali APMC', 'Vadnagar APMC',
                   'Sami APMC', 'Dhari(Chalala) APMC', 'Lakhani APMC', 'Manavdar APMC', 'Vav APMC', 'Anklav APMC',
                   'Borsad(Veg Yard Barsad) APMC', 'Khambhat(Grain Market) APMC', 'Khambhat(Veg Yard Khambhat) APMC',
                   'APMC ANAND', 'Vijapur(Gojjariya) APMC', 'Vijapur(Kukarvada) APMC', 'Vijapur(Vegetable) APMC',
                   'Dehgam(Rekhiyal) APMC', 'Mandal APMC', 'Nadiad APMC', 'Rampura APMC', 'Vansda(Limzar) APMC',
                   'Viramgam APMC', 'Ahwa-Dang APMC', 'Botad(Bhabarkot) APMC', 'Kosamba(Vankal) APMC', 'Kosamba(Zangvav) APMC',
                   'Nadiyad(Piplag) APMC', 'Dasada Patadi APMC', 'Dharampur(veg) APMC', 'Gadada APMC', 'Hasot APMC',
                   'Keshod APMC', 'Pardi APMC', 'Pardi(Motavagachiya) APMC', 'Pardi(Rohinee) APMC', 'Pardi(Ugvada) APMC',
                   'Pardi(Vapi) APMC', 'Ranpur APMC', 'Valia APMC', 'Petlad(Veg Yard, Petlad) APMC', 'Khambhat APMC',
                   'Mehmadabad(Veg ,market) APMC', 'Songadh APMC', 'Valsad(Dungari) APMC', 'Bachau APMC', 'Hasot(Elav) APMC',
                   'Songadh(Badarpada) APMC', 'Bayad APMC', 'Bardoli(Kadod) APMC', 'Bayad(Sadamba) APMC', 'Diyodar APMC',
                   'Kadi(Kadi cotton Yard) APMC', 'Kalol(Veg,Market,Kalol) APMC', 'Limkheda APMC', 'Nadiyad(Chaklasi) APMC',
                   'Porbandar APMC', 'Vankaner(Sub yard) APMC', 'Bhanvad APMC', 'Derol APMC', 'Derol(Adadara) APMC',
                   'Derol(Vejalpur) APMC', 'Idar(Jadar) APMC', 'Mandvi APMC', 'Tarapur APMC', 'Vijapur(Ladol) APMC',
                   'Bayad(Demai) APMC', 'Dholka(Koth) APMC', 'Morva Hafad APMC', 'Gondal(Veg.market Gondal) APMC',
                   'Gogamba(Similiya) APMC', 'Jambusar APMC', 'Valod APMC', 'Kathlal APMC', 'Sayala APMC',
                   'Godhra(Timbaroad) APMC', 'Jambusar(Kaavi) APMC', 'Kodinar(Dollasa) APMC', 'Bardoli(Madhi) APMC',
                   'Jhalod APMC', 'Valia(Nethrang) APMC', 'Khedbrahma(Posina) APMC', 'Gariyadar APMC', 'Savli(Samlaya) APMC',
                   'Damnagar APMC', 'APMC KHAMBHA', 'Shehra APMC', 'Godhra Kakanpur APMC', 'Tharad(Rah) APMC', 'Radhanpur APMC',
                   'Zalod(Sanjeli) APMC', 'Zalod(Zalod) APMC', 'Bilimora(Gandevi) APMC', 'Rajkot(Veg.Sub Yard) APMC',
                   'Dediyapada APMC', 'Unjha APMC', 'Halol APMC', 'Randheja(Chiloda) APMC', 'Bhatiya APMC', 'Panthawada APMC',
                   'Balasinor APMC', 'Songadh(Umrada) APMC', 'Talod(Harsol) APMC', 'Thara(Shihori) APMC',
                   'Baruch Vagara APMC', 'Bavla APMC', 'Randheja(Veg.Market Gandhinagar) APMC', 'Meghraj(Rellavada) APMC',
                   'Jasdan(Vichhiya) APMC', 'Dhragradhra APMC', 'Navsari APMC', 'Veraval APMC', 'Lakhtar APMC',
                   'Vijaynagar(Kundlakap) APMC', 'Chansama APMC', 'Mundra APMC', 'KUKARMUNDA APMC', 'Nizar(Rumkitalav) APMC',
                   'Vaghai APMC', 'Kutiyana APMC', 'JASDAN APMC', 'Khedbrahma(Lambadia) APMC', 'Garudeshwal APMC',
                   'Himatnagar(Gambhoi) APMC', 'Himatnagar(Veg.Market Himatnagar) APMC']

    commodity_list = ['crops', 'seeds', 'pulses', 'vegetables', 'spices', 'fruits', 'leafy_vegetables', 'other', 'oil']


    input_district = st.selectbox(label='Select any district', options=district_list)
    input_market = st.selectbox(label='Select any market', options=market_list)
    input_commodity = st.selectbox(label='Select any commodity', options=commodity_list)
    input_month = st.slider(label='Select Month', min_value=1, max_value=12)

    if st.button(label='Predict'):
        predicton_row = [{
            'district': input_district,
            'market': input_market,
            'commodity_category': input_commodity,
            'months': input_month
        }]

        df = pd.DataFrame(predicton_row)

        production_model = joblib.load('agri_cat_model.joblib')
        raw_preds = production_model.predict(df)
        final_pred_value = max(0.0, float(raw_preds))
        final_pred_value = round(final_pred_value, 2)

        st.write(f'Final prediction for month {input_month} is {final_pred_value} MT.')


if __name__ == "__main__":
    sys.exit(main())