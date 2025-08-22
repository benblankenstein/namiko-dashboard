# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import py7zr
import tempfile
import os
import json
import geopandas as gpd
from sklearn.linear_model import LinearRegression

from shapely.geometry import Point

def modul_mitgliederuebersicht(dfs):
    """Zeigt die vollständige Mitgliederliste an."""
    st.subheader("📋 Mitgliederverzeichnis")
    df = dfs.get('alle_mitglieder')
    if df is None:
        st.warning("Keine Mitgliederdaten vorhanden.")
        return

    # Anzahl Mitglieder anzeigen
    st.markdown(f"**Gesamtzahl Mitglieder:** {len(df)}")

    # Auswahl, wie viele Zeilen angezeigt werden sollen
    n = st.slider("Anzahl der anzuzeigenden Zeilen", min_value=5, max_value=len(df), value=10, step=5)
    st.dataframe(df.head(n))

    # Einzelmitgliedsuche
    st.markdown("---")
    st.markdown("### 🔍 Mitgliedersuche")
    col1, col2 = st.columns(2)
    with col1:
        vorname = st.text_input("Vorname")
    with col2:
        nachname = st.text_input("Nachname")

    if st.button("Suchen"):
        auswahl = df[
            (df['Vorname'].str.lower() == vorname.strip().lower()) &
            (df['Nachname'].str.lower() == nachname.strip().lower())
        ]
        if auswahl.empty:
            st.error(f"❌ Kein Mitglied gefunden mit dem Namen: {vorname} {nachname}")
        else:
            st.dataframe(auswahl)

######################################################################################################################

from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
from datetime import datetime


def modul_geografische_auswertung(dfs):
    import streamlit as st
    from streamlit_folium import st_folium
    import folium
    from folium.plugins import MarkerCluster
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    import time
    from datetime import datetime
    import pandas as pd

    st.subheader("🗺 Geografische Mitgliederübersicht")

    # Prüfen, ob bereits im Session-State gespeichert
    if 'geo_df' in st.session_state:
        dfs['alle_mitglieder_geo'] = st.session_state['geo_df']

    # Wenn keine Geo-Daten in dfs existieren → Nutzer kann Geokodierung starten
    if 'alle_mitglieder_geo' not in dfs:
        st.info("Noch keine Geodaten vorhanden. Bitte Geokodierung starten.")
        if st.button("📍 Geokodierung starten"):
            df_members = dfs.get('alle_mitglieder')
            if df_members is None:
                st.error("Keine Mitgliederdaten verfügbar.")
                return

            df_members['full_address'] = (
                df_members['Straße + Hausnummer'].astype(str) + ", " +
                df_members['PLZ + Ort'].astype(str) + ", Deutschland"
            )

            # Geocoder vorbereiten
            geolocator = Nominatim(user_agent="namiko_geocoder", timeout=10)
            geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2, error_wait_seconds=3)

            def get_coords(address):
                try:
                    location = geocode(address)
                    if location:
                        return pd.Series([location.latitude, location.longitude])
                except:
                    pass
                return pd.Series([None, None])

            st.write("🔄 Starte Geokodierung – dies kann einige Minuten dauern...")
            start_time = time.time()
            df_members[['latitude', 'longitude']] = df_members['full_address'].apply(get_coords)
            end_time = time.time()
            minutes, seconds = divmod(end_time - start_time, 60)

            # In dfs und Session-State speichern
            dfs['alle_mitglieder_geo'] = df_members
            st.session_state['geo_df'] = df_members

            st.success(f"✅ Geokodierung abgeschlossen ({int(minutes)} Minuten {seconds:.1f} Sekunden)")

    else:
        st.info("✅ Bereits vorhandene Geodaten werden verwendet.")

    # Falls nach Geokodierung oder aus Session-State vorhanden → Karte zeichnen
    if 'alle_mitglieder_geo' in dfs:
        df_geo = dfs['alle_mitglieder_geo'].dropna(subset=['latitude', 'longitude'])
        if df_geo.empty:
            st.warning("Keine Mitglieder mit gültigen Koordinaten gefunden.")
            return

        m = folium.Map(location=[51.96236, 7.62571], zoom_start=12)
        marker_cluster = MarkerCluster().add_to(m)
        for _, row in df_geo.iterrows():
            popup_text = f"<b>{row['Vorname']} {row['Nachname']}</b><br>{row['Straße + Hausnummer']}, {row['PLZ + Ort']}"
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=popup_text,
                icon=folium.Icon(color="green", icon="home", prefix="fa")
            ).add_to(marker_cluster)

        st_folium(m, width=700, height=500)


######################################################################################################################

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns  # Behalten für die Farbpalette, falls gewünscht
import ast


# Die Hilfsfunktion wird hier ebenfalls benötigt
def count_products(products):
    if isinstance(products, str):
        try:
            products = ast.literal_eval(products)
        except (ValueError, SyntaxError):
            return 0
    if isinstance(products, list):
        return len(products)
    return 0


def modul_produktkategorien(dfs):
    st.subheader("📦 Analyse der Produktkategorien")

    # 1. Daten-Prüfung und -Vorbereitung
    # =================================================================
    if 'product_categories' not in dfs:
        st.warning("Benötigte Daten ('product_categories') nicht gefunden.")
        return

    df_categories = dfs['product_categories'].copy()
    df_categories['product_count'] = df_categories['products'].apply(count_products)

    # Für die Visualisierung sortieren
    df_plot_cat = df_categories.sort_values(by='product_count', ascending=False)

    # 2. Visualisierung in Streamlit
    # =================================================================
    st.markdown("#### Anzahl der Produkte pro Kategorie")

    fig = px.bar(
        df_plot_cat,
        x='name',
        y='product_count',
        title="Anzahl der Produkte pro Produktkategorie",
        labels={'name': 'Produktkategorie', 'product_count': 'Anzahl Produkte'},
        color='product_count',
        color_continuous_scale=px.colors.sequential.Greens_r,
        text_auto=True  # Zeigt die Werte direkt auf den Balken an
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

######################################################################################################################

import streamlit as st
import pandas as pd
import plotly.express as px
import ast
from datetime import datetime


# Die Hilfsfunktion aus deinem Code
def count_products(products):
    if isinstance(products, str):
        try:
            products = ast.literal_eval(products)
        except (ValueError, SyntaxError):
            return 0
    if isinstance(products, list):
        return len(products)
    return 0


import streamlit as st
import pandas as pd
import plotly.express as px
import ast
from datetime import datetime


# Die Hilfsfunktion bleibt unverändert
def count_products(products):
    if isinstance(products, str):
        try:
            products = ast.literal_eval(products)
        except (ValueError, SyntaxError):
            return 0
    if isinstance(products, list):
        return len(products)
    return 0

######################################################################################################################

import streamlit as st
import pandas as pd
import plotly.express as px
import ast
from datetime import datetime


# Die Hilfsfunktion bleibt unverändert
def count_products(products):
    if isinstance(products, str):
        try:
            products = ast.literal_eval(products)
        except (ValueError, SyntaxError):
            return 0
    if isinstance(products, list):
        return len(products)
    return 0


def modul_lieferantenanalyse(dfs):
    st.subheader("🚚 Lieferantenanalyse: Produktanzahl vs. Aktualität")

    # 1. Daten-Prüfung und -Vorbereitung (unverändert)
    # =================================================================
    if 'suppliers' not in dfs or 'order_items' not in dfs:
        st.warning("Benötigte Daten ('suppliers', 'order_items') nicht gefunden.")
        return

    # Product -> Supplier Mapping
    product_to_supplier = {}
    for _, row in dfs['suppliers'].iterrows():
        try:
            products = row['products']
            if isinstance(products, str):
                products = ast.literal_eval(products)
            if isinstance(products, list):
                for prod in products:
                    if prod and "id" in prod:
                        product_to_supplier[prod.get("id")] = row['name']
        except Exception:
            continue

    # Order-Items + Supplier
    df_items = dfs['order_items'][['product_id_id', 'date_created']].copy()
    df_items['date_created'] = pd.to_datetime(df_items['date_created'], errors='coerce')
    df_items['supplier'] = df_items['product_id_id'].map(product_to_supplier)

    # Letztes Bestelldatum pro Supplier
    df_last_order = (
        df_items.dropna(subset=['supplier', 'date_created'])
        .groupby('supplier')['date_created']
        .max()
        .reset_index()
    )
    if not df_last_order.empty:
        df_last_order['date_created'] = df_last_order['date_created'].dt.tz_localize(None)
        df_last_order['days_since_last_order'] = (datetime.now() - df_last_order['date_created']).dt.days

    # Supplier-Daten
    df_suppliers = dfs['suppliers'].copy()
    df_suppliers['product_count'] = df_suppliers['products'].apply(count_products)

    # Merge für Plot
    df_plot = df_suppliers[['name', 'product_count']].merge(
        df_last_order[['supplier', 'days_since_last_order']],
        left_on='name',
        right_on='supplier',
        how='left'
    ).drop(columns='supplier')

    df_plot['days_since_last_order'] = df_plot['days_since_last_order'].fillna(-1)
    df_plot = df_plot.sort_values('name').reset_index(drop=True)
    df_plot['number'] = df_plot.index + 1

    # 2. Visualisierung in Streamlit (angepasst)
    # =================================================================
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Interaktiver Scatter-Plot")
        fig_scatter = px.scatter(
            df_plot,
            x='product_count',
            y='days_since_last_order',
            hover_name='name',
            hover_data={'product_count': True, 'days_since_last_order': True, 'number': False},
            text='number',
            title="Produktanzahl vs. Aktualität",
            labels={
                'product_count': 'Anzahl der Produkte',
                'days_since_last_order': 'Tage seit letzter Bestellung'
            }
        )

        # Styling der Punkte und des Textes
        fig_scatter.update_traces(
            marker=dict(size=12, color='#4c9e6a'),  # **NEU**: Größe der Punkte auf 12 erhöht
            textposition='top center',
            textfont_size=10
        )
        fig_scatter.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        st.markdown("#### Legende")
        # **NEU**: Feste Höhe entfernt, Tabelle passt sich dem Inhalt an
        st.dataframe(
            df_plot[['number', 'name']],
            hide_index=True,
            use_container_width=True
        )


    # 4. Optionale Säulendiagramme (unverändert)
    # =================================================================
    if st.checkbox("Zusätzliche Säulendiagramme anzeigen"):
        df_sorted_count = df_plot.sort_values('product_count', ascending=False)
        df_sorted_days = df_plot.sort_values('days_since_last_order', ascending=True)

        bar_col1, bar_col2 = st.columns(2)
        with bar_col1:
            st.markdown("##### Anzahl Produkte")
            fig_bar_count = px.bar(df_sorted_count.head(20), x='name', y='product_count')
            fig_bar_count.update_traces(marker_color='#4c9e6a')
            st.plotly_chart(fig_bar_count, use_container_width=True)

        with bar_col2:
            st.markdown("##### Tage seit letzter Bestellung")
            fig_bar_days = px.bar(df_sorted_days[df_sorted_days['days_since_last_order'] != -1].head(20), x='name',
                                  y='days_since_last_order')
            fig_bar_days.update_traces(marker_color='#4c9e6a')
            st.plotly_chart(fig_bar_days, use_container_width=True)

######################################################################################################################

### Modul 5: Analyse der Solidaritätssteuer ```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def modul_solidaritaetssteuer(dfs):
    st.subheader("💶 Analyse der Solidaritätssteuer")

    # 1. Daten-Prüfung und -Vorbereitung (exakt nach deinem Notebook)
    # =================================================================
    if 'balance_sheet' not in dfs:
        st.warning("Benötigte Daten ('balance_sheet') nicht gefunden.")
        return

    df_bs = dfs['balance_sheet'].copy()
    valid_tax_values = {0.0, 0.01, 0.03, 0.05}

    # Numerische Umwandlung sicherstellen
    for col in ['product_total', 'total', 'solidarity_tax']:
        df_bs[col] = pd.to_numeric(df_bs[col], errors='coerce')
    df_bs.dropna(subset=['product_total', 'total'], inplace=True)

    # Korrektur der Steuerwerte
    mask_invalid = ~df_bs['solidarity_tax'].isin(valid_tax_values) & df_bs['total'].notna() & (df_bs['total'] != 0)
    df_bs.loc[mask_invalid, 'solidarity_tax'] = (
        1 - df_bs.loc[mask_invalid, 'product_total'] / df_bs.loc[mask_invalid, 'total']
    ).round(2)

    # Neue Spalte 'solidarity_amount' berechnen
    df_bs['solidarity_amount'] = df_bs['total'] - df_bs['product_total']

    # Binning für Produktumsatz
    p95_product = df_bs['product_total'].quantile(0.95)
    bins_product = list(np.arange(0, p95_product, 5)) + [df_bs['product_total'].max()]
    df_bs['product_interval'] = pd.cut(df_bs['product_total'], bins=bins_product, right=False, include_lowest=True)
    prod_labels = list(df_bs['product_interval'].cat.categories.astype(str))
    if prod_labels:
        prod_labels[-1] = f"Top 5% (>= {p95_product:.2f} €)"
        df_bs['product_interval'] = df_bs['product_interval'].cat.rename_categories(prod_labels)

    # Binning für Solidaritätsbeitrag
    p95_tax = df_bs['solidarity_amount'].quantile(0.95)
    max_tax = df_bs['solidarity_amount'].max()
    bins_tax = [0, 0.0000001, 0.5] + list(np.arange(0.5, p95_tax, 0.5)) + [max_tax]
    bins_tax = sorted(set(bins_tax))
    df_bs['tax_interval'] = pd.cut(df_bs['solidarity_amount'], bins=bins_tax, right=False, include_lowest=True, duplicates='drop')
    tax_labels = list(df_bs['tax_interval'].cat.categories.astype(str))
    if tax_labels:
        tax_labels[0] = "0 €"
        tax_labels[-1] = f"Top 5% (>= {p95_tax:.2f} €)"
        df_bs['tax_interval'] = df_bs['tax_interval'].cat.rename_categories(tax_labels)

    # Häufigkeiten für Plots
    product_counts = df_bs['product_interval'].value_counts().sort_index()
    tax_counts = df_bs['tax_interval'].value_counts().sort_index()
    tax_pie_counts = df_bs['solidarity_tax'].value_counts().reindex([0.0, 0.01, 0.03, 0.05], fill_value=0)

    # 2. Visualisierungen in Streamlit
    # =================================================================
    green_palette = ['#43a047', '#66bb6a', '#81c784', '#4caf50', '#388e3c']

    st.markdown("#### Verteilung der Warenkorbwerte")
    col1, col2 = st.columns(2)
    with col1:
        fig_bar_prod = px.bar(
            x=product_counts.index.astype(str),
            y=product_counts.values,
            labels={'x': 'Betragsintervall (€)', 'y': 'Anzahl Bestellungen'},
            color_discrete_sequence=[green_palette[2]]
        )
        st.plotly_chart(fig_bar_prod, use_container_width=True)
    with col2:
        fig_box_prod = px.box(
            df_bs,
            y='product_total',
            labels={'product_total': 'Betrag (€)'},
            color_discrete_sequence=[green_palette[0]],
            points=False
        )
        st.plotly_chart(fig_box_prod, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Verteilung der Solidaritätsbeiträge")
    col3, col4 = st.columns(2)
    with col3:
        fig_bar_tax = px.bar(
            x=tax_counts.index.astype(str),
            y=tax_counts.values,
            labels={'x': 'Betragsintervall (€)', 'y': 'Anzahl Bestellungen'},
            color_discrete_sequence=[green_palette[2]]
        )
        st.plotly_chart(fig_bar_tax, use_container_width=True)
    with col4:
        fig_box_tax = px.box(
            df_bs,
            y='solidarity_amount',
            labels={'solidarity_amount': 'Betrag (€)'},
            color_discrete_sequence=[green_palette[0]],
            points=False
        )
        st.plotly_chart(fig_box_tax, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Verteilung der Solidaritätssteuersätze")
    pie_labels = ["0%", "1%", "3%", "5%"]
    fig_pie = px.pie(
        values=tax_pie_counts.values,
        names=pie_labels,
        title="Gewählte Steuersätze",
        color_discrete_sequence=green_palette,
        hole=0.3
    )
    fig_pie.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)



######################################################################################################################

### Modul 5: Analyse der Solidaritätssteuer ```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def modul_solidaritaetssteuer(dfs):
    st.subheader("💶 Analyse der Solidaritätssteuer")

    # 1. Daten-Prüfung und -Vorbereitung (exakt nach deinem Notebook)
    # =================================================================
    if 'balance_sheet' not in dfs:
        st.warning("Benötigte Daten ('balance_sheet') nicht gefunden.")
        return

    df_bs = dfs['balance_sheet'].copy()
    valid_tax_values = {0.0, 0.01, 0.03, 0.05}

    # Numerische Umwandlung sicherstellen
    for col in ['product_total', 'total', 'solidarity_tax']:
        df_bs[col] = pd.to_numeric(df_bs[col], errors='coerce')
    df_bs.dropna(subset=['product_total', 'total'], inplace=True)

    # Korrektur der Steuerwerte
    mask_invalid = ~df_bs['solidarity_tax'].isin(valid_tax_values) & df_bs['total'].notna() & (df_bs['total'] != 0)
    df_bs.loc[mask_invalid, 'solidarity_tax'] = (
        1 - df_bs.loc[mask_invalid, 'product_total'] / df_bs.loc[mask_invalid, 'total']
    ).round(2)

    # Neue Spalte 'solidarity_amount' berechnen
    df_bs['solidarity_amount'] = df_bs['total'] - df_bs['product_total']

    # Binning für Produktumsatz
    p95_product = df_bs['product_total'].quantile(0.95)
    bins_product = list(np.arange(0, p95_product, 5)) + [df_bs['product_total'].max()]
    df_bs['product_interval'] = pd.cut(df_bs['product_total'], bins=bins_product, right=False, include_lowest=True)
    prod_labels = list(df_bs['product_interval'].cat.categories.astype(str))
    if prod_labels:
        prod_labels[-1] = f"Top 5% (>= {p95_product:.2f} €)"
        df_bs['product_interval'] = df_bs['product_interval'].cat.rename_categories(prod_labels)

    # Binning für Solidaritätsbeitrag
    p95_tax = df_bs['solidarity_amount'].quantile(0.95)
    max_tax = df_bs['solidarity_amount'].max()
    bins_tax = [0, 0.0000001, 0.5] + list(np.arange(0.5, p95_tax, 0.5)) + [max_tax]
    bins_tax = sorted(set(bins_tax))
    df_bs['tax_interval'] = pd.cut(df_bs['solidarity_amount'], bins=bins_tax, right=False, include_lowest=True, duplicates='drop')
    tax_labels = list(df_bs['tax_interval'].cat.categories.astype(str))
    if tax_labels:
        tax_labels[0] = "0 €"
        tax_labels[-1] = f"Top 5% (>= {p95_tax:.2f} €)"
        df_bs['tax_interval'] = df_bs['tax_interval'].cat.rename_categories(tax_labels)

    # Häufigkeiten für Plots
    product_counts = df_bs['product_interval'].value_counts().sort_index()
    tax_counts = df_bs['tax_interval'].value_counts().sort_index()
    tax_pie_counts = df_bs['solidarity_tax'].value_counts().reindex([0.0, 0.01, 0.03, 0.05], fill_value=0)

    # 2. Visualisierungen in Streamlit
    # =================================================================
    green_palette = ['#43a047', '#66bb6a', '#81c784', '#4caf50', '#388e3c']

    st.markdown("#### Verteilung der Warenkorbwerte")
    col1, col2 = st.columns(2)
    with col1:
        fig_bar_prod = px.bar(
            x=product_counts.index.astype(str),
            y=product_counts.values,
            labels={'x': 'Betragsintervall (€)', 'y': 'Anzahl Bestellungen'},
            color_discrete_sequence=[green_palette[2]]
        )
        st.plotly_chart(fig_bar_prod, use_container_width=True)
    with col2:
        fig_box_prod = px.box(
            df_bs,
            y='product_total',
            labels={'product_total': 'Betrag (€)'},
            color_discrete_sequence=[green_palette[0]],
            points=False
        )
        st.plotly_chart(fig_box_prod, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Verteilung der Solidaritätsbeiträge")
    col3, col4 = st.columns(2)
    with col3:
        fig_bar_tax = px.bar(
            x=tax_counts.index.astype(str),
            y=tax_counts.values,
            labels={'x': 'Betragsintervall (€)', 'y': 'Anzahl Bestellungen'},
            color_discrete_sequence=[green_palette[2]]
        )
        st.plotly_chart(fig_bar_tax, use_container_width=True)
    with col4:
        fig_box_tax = px.box(
            df_bs,
            y='solidarity_amount',
            labels={'solidarity_amount': 'Betrag (€)'},
            color_discrete_sequence=[green_palette[0]],
            points=False
        )
        st.plotly_chart(fig_box_tax, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Verteilung der Solidaritätssteuersätze")
    pie_labels = ["0%", "1%", "3%", "5%"]
    fig_pie = px.pie(
        values=tax_pie_counts.values,
        names=pie_labels,
        title="Gewählte Steuersätze",
        color_discrete_sequence=green_palette,
        hole=0.3
    )
    fig_pie.update_traces(textinfo='percent+label')
    st.plotly_chart(fig_pie, use_container_width=True)


######################################################################################################################

def modul_bestellhaeufigkeit(dfs):
    st.subheader("📈 Analyse der Bestellhäufigkeit pro Mitglied")

    # 1. Daten-Prüfung und -Vorbereitung
    # =================================================================
    if 'balance_sheet' not in dfs:
        st.warning("Benötigte Daten ('balance_sheet') nicht gefunden.")
        return

    order_counts = dfs['balance_sheet']['customer_name'].value_counts()

    # Die Werte-Spalte wird 'Anzahl Mitglieder', die Index-Spalte heißt 'count'.
    freq_dist = order_counts.value_counts().sort_index().reset_index(name='Anzahl Mitglieder')

    # *** HIER IST DIE FINALE KORREKTUR ***
    # Wir benennen die Spalte 'count' (die die Anzahl der Bestellungen enthält) korrekt um.
    freq_dist = freq_dist.rename(columns={'count': 'Anzahl Bestellungen'})

    # 2. Visualisierung in Streamlit
    # =================================================================
    fig = px.bar(
        freq_dist,
        x='Anzahl Bestellungen',
        y='Anzahl Mitglieder',
        title="Anzahl Mitglieder je Bestellhäufigkeit",
        text_auto=True,
        color_discrete_sequence=['#43a047']
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    # 3. Auswertungsbox
    #

######################################################################################################################

# Hilfsfunktion, um eine Liste aus einem String sicher zu evaluieren
def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []  # Leere Liste bei Fehler


def modul_top_flop(dfs):
    st.subheader("🚀 Top & Flop Analyse")

    # 1. Daten-Prüfung und -Vorbereitung
    # =================================================================
    required_dfs = ['order_items', 'products', 'product_categories']
    if not all(df in dfs for df in required_dfs):
        st.warning("Benötigte Daten ('order_items', 'products', 'product_categories') nicht gefunden.")
        return

    # Kopien erstellen und Datentypen sicherstellen
    order_items = dfs['order_items'].copy()
    products = dfs['products'].copy()
    categories = dfs['product_categories'].copy()

    order_items['quantity'] = pd.to_numeric(order_items['quantity'], errors='coerce').fillna(0)
    products['price_per_unit'] = pd.to_numeric(products['price_per_unit'], errors='coerce').fillna(0)
    products['tax_rate'] = pd.to_numeric(products['tax_rate'], errors='coerce').fillna(0)

    # 2. Interaktive Steuerelemente für den User
    # =================================================================
    st.markdown("##### Analyseeinstellungen")
    col1, col2, col3 = st.columns(3)
    with col1:
        analysis_type = st.radio("Analyseart", ["Produkte", "Kategorien"], key="analysis_type")
    with col2:
        metric_type = st.radio("Metrik", ["Menge", "Warenwert", "Beide"], key="metric_type")
    with col3:
        display_type = st.radio("Darstellung", ["Tabelle", "Diagramm"], key="display_type")

    # 3. Logik für die Produkt-Analyse
    # =================================================================
    if analysis_type == "Produkte":
        # Daten zusammenführen
        merged = order_items.groupby(['product_id_unit', 'product_id_id'], as_index=False)['quantity'].sum()
        merged = merged.merge(
            products[['id', 'name', 'price_per_unit', 'tax_rate']],
            left_on='product_id_id', right_on='id', how='left'
        )
        merged['gross_value'] = merged['quantity'] * merged['price_per_unit'] * (1 + merged['tax_rate'])

        einheiten = sorted(merged['product_id_unit'].dropna().unique())
        selected_unit = st.selectbox("Wähle eine Einheit zur Analyse:", einheiten)

        if selected_unit:
            df_unit = merged[merged['product_id_unit'] == selected_unit]

            # Analyse nach MENGE
            if metric_type in ["Menge", "Beide"]:
                st.markdown(f"#### Analyse nach Menge für Einheit: `{selected_unit}`")
                top3_qty = df_unit.nlargest(3, 'quantity')
                flop3_qty = df_unit.nsmallest(3, 'quantity')

                if display_type == "Tabelle":
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("Top 3")
                        st.dataframe(
                            top3_qty[['name', 'quantity']].rename(columns={'name': 'Name', 'quantity': 'Menge'}))
                    with c2:
                        st.write("Flop 3")
                        st.dataframe(
                            flop3_qty[['name', 'quantity']].rename(columns={'name': 'Name', 'quantity': 'Menge'}))
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = px.bar(top3_qty, x='quantity', y='name', orientation='h', title="Top 3 nach Menge",
                                     text_auto=True, color_discrete_sequence=['#43a047'])
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        fig = px.bar(flop3_qty, x='quantity', y='name', orientation='h', title="Flop 3 nach Menge",
                                     text_auto=True, color_discrete_sequence=['#81c784'])
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)

            # Analyse nach WARENWERT
            if metric_type in ["Warenwert", "Beide"]:
                st.markdown(f"#### Analyse nach Warenwert für Einheit: `{selected_unit}`")
                top3_val = df_unit.nlargest(3, 'gross_value')
                flop3_val = df_unit.nsmallest(3, 'gross_value')

                if display_type == "Tabelle":
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("Top 3")
                        st.dataframe(top3_val[['name', 'gross_value']].rename(
                            columns={'name': 'Name', 'gross_value': 'Warenwert (€)'}))
                    with c2:
                        st.write("Flop 3")
                        st.dataframe(flop3_val[['name', 'gross_value']].rename(
                            columns={'name': 'Name', 'gross_value': 'Warenwert (€)'}))
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = px.bar(top3_val, x='gross_value', y='name', orientation='h', title="Top 3 nach Warenwert",
                                     text_auto='.2f', color_discrete_sequence=['#43a047'])
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        fig = px.bar(flop3_val, x='gross_value', y='name', orientation='h',
                                     title="Flop 3 nach Warenwert", text_auto='.2f',
                                     color_discrete_sequence=['#81c784'])
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)

    # 4. Logik für die Kategorie-Analyse
    # =================================================================
    elif analysis_type == "Kategorien":
        # Kategorien-Mapping erstellen (explodieren)
        categories['products'] = categories['products'].apply(safe_literal_eval)
        cat_mapping = categories.explode('products').rename(columns={'products': 'product_id', 'name': 'category'})

        # Daten zusammenführen
        merged_cat = order_items.merge(products[['id', 'price_per_unit', 'tax_rate']], left_on='product_id_id',
                                       right_on='id')
        merged_cat['gross_value'] = merged_cat['quantity'] * merged_cat['price_per_unit'] * (1 + merged_cat['tax_rate'])
        merged_cat = merged_cat.merge(cat_mapping[['category', 'product_id']], left_on='product_id_id',
                                      right_on='product_id')

        # Analyse nach MENGE
        if metric_type in ["Menge", "Beide"]:
            st.markdown("#### Top 3 Kategorien nach Menge")
            top_cat_qty = merged_cat.groupby('category')['quantity'].sum().nlargest(3).reset_index()
            if display_type == "Tabelle":
                st.dataframe(top_cat_qty)
            else:
                fig = px.bar(top_cat_qty, x='quantity', y='category', orientation='h',
                             title="Top 3 Kategorien nach Menge", text_auto=True, color_discrete_sequence=['#43a047'])
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

        # Analyse nach WARENWERT
        if metric_type in ["Warenwert", "Beide"]:
            st.markdown("#### Top 3 Kategorien nach Warenwert")
            top_cat_val = merged_cat.groupby('category')['gross_value'].sum().nlargest(3).reset_index()
            if display_type == "Tabelle":
                st.dataframe(top_cat_val.rename(columns={'gross_value': 'Warenwert (€)'}))
            else:
                fig = px.bar(top_cat_val, x='gross_value', y='category', orientation='h',
                             title="Top 3 Kategorien nach Warenwert", text_auto='.2f',
                             color_discrete_sequence=['#43a047'])
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

######################################################################################################################


def modul_wochentage(dfs):
    st.subheader("🗓️ Bestellungen nach Wochentagen")

    # 1. Daten-Prüfung und -Vorbereitung
    # =================================================================
    if 'order_items' not in dfs:
        st.warning("Benötigte Daten ('order_items') nicht gefunden.")
        return

    order_items = dfs['order_items'].copy()
    order_items['date_created'] = pd.to_datetime(order_items['date_created'], errors='coerce')
    order_items.dropna(subset=['date_created'], inplace=True)

    # 2. Interaktives Steuerelement
    # =================================================================
    block_size = 3

    # 3. Berechnungslogik (aus dem Notebook übernommen)
    # =================================================================
    # Wochen mit Bestellungen identifizieren
    weekly_counts = order_items.set_index('date_created').resample('W-Mon').size()
    active_weeks = weekly_counts[weekly_counts > 0].index

    if active_weeks.empty:
        st.warning("Keine Bestelldaten für die Analyse gefunden.")
        return

    # Blöcke aus aktiven Wochen bilden
    num_blocks = (len(active_weeks) + block_size - 1) // block_size

    weekday_summaries = []
    for i in range(num_blocks):
        block_start_week = active_weeks[i * block_size]
        # Das Ende des Blocks ist der Start + block_size Wochen, aber nicht über das Ende der aktiven Wochen hinaus
        end_idx = min((i + 1) * block_size - 1, len(active_weeks) - 1)
        block_end_week = active_weeks[end_idx]

        mask = (order_items['date_created'].dt.to_period('W-Mon') >= block_start_week.to_period('W-Mon')) & \
               (order_items['date_created'].dt.to_period('W-Mon') <= block_end_week.to_period('W-Mon'))

        block_rows = order_items.loc[mask]

        if not block_rows.empty:
            counts = block_rows.groupby(block_rows['date_created'].dt.weekday).size().reindex(range(7), fill_value=0)
            weekday_summaries.append(counts)

    if not weekday_summaries:
        st.warning("Konnte keine Analyse-Perioden erstellen. Versuche eine andere Blockgröße.")
        return

    # Durchschnitt über alle Blöcke berechnen
    weekday_avg = pd.concat(weekday_summaries, axis=1).mean(axis=1)
    weekday_avg.index = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']

    # 4. Visualisierung
    # =================================================================
    fig = px.bar(
        weekday_avg,
        x=weekday_avg.index,
        y=weekday_avg.values,
        title=f"Durchschnittlich bestellte Produkte pro Wochentag",
        labels={'x': 'Wochentag', 'y': 'Durchschnittliche Anzahl Produkte'},
        text_auto='.2f',
        color_discrete_sequence=['#43a047']
    )
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

#####################################################################################################################

# Hilfsfunktion, um eine Liste aus einem String sicher zu evaluieren
def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []  # Leere Liste bei Fehler


def modul_top_flop(dfs):
    st.subheader("🚀 Top & Flop Analyse")

    # 1. Daten-Prüfung und -Vorbereitung
    # =================================================================
    required_dfs = ['order_items', 'products', 'product_categories']
    if not all(df in dfs for df in required_dfs):
        st.warning("Benötigte Daten ('order_items', 'products', 'product_categories') nicht gefunden.")
        return

    # Kopien erstellen und Datentypen sicherstellen
    order_items = dfs['order_items'].copy()
    products = dfs['products'].copy()
    categories = dfs['product_categories'].copy()

    order_items['quantity'] = pd.to_numeric(order_items['quantity'], errors='coerce').fillna(0)
    products['price_per_unit'] = pd.to_numeric(products['price_per_unit'], errors='coerce').fillna(0)
    products['tax_rate'] = pd.to_numeric(products['tax_rate'], errors='coerce').fillna(0)

    # 2. Interaktive Steuerelemente für den User
    # =================================================================
    st.markdown("##### Analyseeinstellungen")
    col1, col2, col3 = st.columns(3)
    with col1:
        analysis_type = st.radio("Analyseart", ["Produkte", "Kategorien"], key="analysis_type")
    with col2:
        metric_type = st.radio("Metrik", ["Menge", "Warenwert", "Beide"], key="metric_type")
    with col3:
        display_type = st.radio("Darstellung", ["Tabelle", "Diagramm"], key="display_type")

    # 3. Logik für die Produkt-Analyse
    # =================================================================
    if analysis_type == "Produkte":
        # Daten zusammenführen
        merged = order_items.groupby(['product_id_unit', 'product_id_id'], as_index=False)['quantity'].sum()
        merged = merged.merge(
            products[['id', 'name', 'price_per_unit', 'tax_rate']],
            left_on='product_id_id', right_on='id', how='left'
        )
        merged['gross_value'] = merged['quantity'] * merged['price_per_unit'] * (1 + merged['tax_rate'])

        einheiten = sorted(merged['product_id_unit'].dropna().unique())
        selected_unit = st.selectbox("Wähle eine Einheit zur Analyse:", einheiten)

        if selected_unit:
            df_unit = merged[merged['product_id_unit'] == selected_unit]

            # Analyse nach MENGE
            if metric_type in ["Menge", "Beide"]:
                st.markdown(f"#### Analyse nach Menge für Einheit: `{selected_unit}`")
                top3_qty = df_unit.nlargest(3, 'quantity')
                flop3_qty = df_unit.nsmallest(3, 'quantity')

                if display_type == "Tabelle":
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("Top 3")
                        st.dataframe(
                            top3_qty[['name', 'quantity']].rename(columns={'name': 'Name', 'quantity': 'Menge'}))
                    with c2:
                        st.write("Flop 3")
                        st.dataframe(
                            flop3_qty[['name', 'quantity']].rename(columns={'name': 'Name', 'quantity': 'Menge'}))
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = px.bar(top3_qty, x='quantity', y='name', orientation='h', title="Top 3 nach Menge",
                                     text_auto=True, color_discrete_sequence=['#43a047'])
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        fig = px.bar(flop3_qty, x='quantity', y='name', orientation='h', title="Flop 3 nach Menge",
                                     text_auto=True, color_discrete_sequence=['#81c784'])
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)

            # Analyse nach WARENWERT
            if metric_type in ["Warenwert", "Beide"]:
                st.markdown(f"#### Analyse nach Warenwert für Einheit: `{selected_unit}`")
                top3_val = df_unit.nlargest(3, 'gross_value')
                flop3_val = df_unit.nsmallest(3, 'gross_value')

                if display_type == "Tabelle":
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write("Top 3")
                        st.dataframe(top3_val[['name', 'gross_value']].rename(
                            columns={'name': 'Name', 'gross_value': 'Warenwert (€)'}))
                    with c2:
                        st.write("Flop 3")
                        st.dataframe(flop3_val[['name', 'gross_value']].rename(
                            columns={'name': 'Name', 'gross_value': 'Warenwert (€)'}))
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = px.bar(top3_val, x='gross_value', y='name', orientation='h', title="Top 3 nach Warenwert",
                                     text_auto='.2f', color_discrete_sequence=['#43a047'])
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        fig = px.bar(flop3_val, x='gross_value', y='name', orientation='h',
                                     title="Flop 3 nach Warenwert", text_auto='.2f',
                                     color_discrete_sequence=['#81c784'])
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)

    # 4. Logik für die Kategorie-Analyse
    # =================================================================
    elif analysis_type == "Kategorien":
        # Kategorien-Mapping erstellen (explodieren)
        categories['products'] = categories['products'].apply(safe_literal_eval)
        cat_mapping = categories.explode('products').rename(columns={'products': 'product_id', 'name': 'category'})

        # Daten zusammenführen
        merged_cat = order_items.merge(products[['id', 'price_per_unit', 'tax_rate']], left_on='product_id_id',
                                       right_on='id')
        merged_cat['gross_value'] = merged_cat['quantity'] * merged_cat['price_per_unit'] * (1 + merged_cat['tax_rate'])
        merged_cat = merged_cat.merge(cat_mapping[['category', 'product_id']], left_on='product_id_id',
                                      right_on='product_id')

        # Analyse nach MENGE
        if metric_type in ["Menge", "Beide"]:
            st.markdown("#### Top 3 Kategorien nach Menge")
            top_cat_qty = merged_cat.groupby('category')['quantity'].sum().nlargest(3).reset_index()
            if display_type == "Tabelle":
                st.dataframe(top_cat_qty)
            else:
                fig = px.bar(top_cat_qty, x='quantity', y='category', orientation='h',
                             title="Top 3 Kategorien nach Menge", text_auto=True, color_discrete_sequence=['#43a047'])
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

        # Analyse nach WARENWERT
        if metric_type in ["Warenwert", "Beide"]:
            st.markdown("#### Top 3 Kategorien nach Warenwert")
            top_cat_val = merged_cat.groupby('category')['gross_value'].sum().nlargest(3).reset_index()
            if display_type == "Tabelle":
                st.dataframe(top_cat_val.rename(columns={'gross_value': 'Warenwert (€)'}))
            else:
                fig = px.bar(top_cat_val, x='gross_value', y='category', orientation='h',
                             title="Top 3 Kategorien nach Warenwert", text_auto='.2f',
                             color_discrete_sequence=['#43a047'])
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

######################################################################################################################

def build_order_blocks(order_items, block_size=3):
    df = order_items.copy()


    df['date_created'] = pd.to_datetime(df['date_created'], errors='coerce')
    df = df.dropna(subset=['date_created']).set_index('date_created')

    weekly = df.resample('W-Mon').size().reset_index(name='dummy')  # W-Mon for Sunday start

    grouped_blocks = []
    current_block = []
    for _, row in weekly.iterrows():
        if row['dummy'] > 0:
            current_block.append(row)
            if len(current_block) == block_size:
                start_date = current_block[0]['date_created']
                end_date = current_block[-1]['date_created']
                mid_date = start_date + (end_date - start_date) / 2
                grouped_blocks.append({'start_date': start_date, 'end_date': end_date, 'date_created': mid_date})
                current_block = []
        else:
            if current_block:
                start_date = current_block[0]['date_created']
                end_date = current_block[-1]['date_created']
                mid_date = start_date + (end_date - start_date) / 2
                grouped_blocks.append({'start_date': start_date, 'end_date': end_date, 'date_created': mid_date})
                current_block = []

    if current_block:
        start_date = current_block[0]['date_created']
        end_date = current_block[-1]['date_created']
        mid_date = start_date + (end_date - start_date) / 2
        grouped_blocks.append({'start_date': start_date, 'end_date': end_date, 'date_created': mid_date})

    df_blocks = pd.DataFrame(grouped_blocks)
    df_blocks['year'] = df_blocks['date_created'].dt.year
    df_blocks['month_year'] = df_blocks['date_created'].dt.strftime('%b %Y')
    return df_blocks


def modul_bestellbloecke(dfs):
    st.subheader("📦 Analyse der Bestellblöcke")


    # 1. Daten-Prüfung und -Vorbereitung
    # =================================================================
    if 'order_items' not in dfs or 'products' not in dfs:
        st.warning("Benötigte Daten ('order_items', 'products') nicht gefunden.")
        return

    order_items = dfs['order_items'].copy()
    order_items['date_created'] = pd.to_datetime(order_items['date_created'], errors='coerce')

    # 2. Interaktives Steuerelement
    # =================================================================
    selected_metric = st.radio(
        "Welche Metrik möchtest du anzeigen?",
        ('Warenwert', 'Mitglieder', 'Nicht bestellte Produkte', 'Alle'),
        horizontal=True
    )

    # 3. Block-Template und Plot-Funktion definieren
    # =================================================================
    block_template = build_order_blocks(order_items, block_size=3)


    def plot_metric(metric_key):
        # --- Datenaufbereitung je nach Metrik ---
        if metric_key == 'revenue':
            products = dfs['products'].copy()
            data = order_items.merge(products[['id', 'price_per_unit', 'tax_rate']], left_on='product_id_id', right_on='id')
            data['quantity'] = pd.to_numeric(data['quantity'], errors='coerce').fillna(0)
            data['gross_value'] = data['quantity'] * data['price_per_unit'] * (1 + data['tax_rate'])
            weekly_data = data.set_index('date_created').resample('W-Mon')['gross_value'].sum().reset_index(name='value')
            y_label, title = "Warenwert (€)", "Warenwert pro Bestellblock"

        elif metric_key == 'customers':
            data = order_items.copy()
            data['customer_id'] = data['order_id_customer_first_name'].str.strip() + "_" + data[
                'order_id_customer_last_name'].str.strip()
            weekly_data = data.groupby(pd.Grouper(key='date_created', freq='W-Mon'))['customer_id'].nunique().reset_index(
                name='value')
            y_label, title = "Anzahl Mitglieder", "Mitglieder pro Bestellblock"

        elif metric_key == 'not_ordered':
            data = order_items.copy()
            data['not_ordered'] = (
                        data['order_id_customer_first_name'].isna() & data['order_id_customer_last_name'].isna()).astype(
                int)
            weekly_data = data.groupby(pd.Grouper(key='date_created', freq='W-Mon'))['not_ordered'].sum().reset_index(
                name='value')
            y_label, title = "Anzahl Produkte", "Nicht bestellte Produkte pro Bestellblock"

        weekly_data = weekly_data.set_index('date_created')

        # --- Daten mit Block-Template zusammenführen ---
        block_values = []
        for _, block in block_template.iterrows():
            mask = (weekly_data.index >= block['start_date']) & (weekly_data.index <= block['end_date'])
            block_sum = weekly_data.loc[mask, 'value'].sum() if mask.any() else 0
            block_values.append({'month_year': block['month_year'], 'year': block['year'], 'value': block_sum})

        df_plot = pd.DataFrame(block_values)

        # --- Plot erstellen ---
        fig = px.bar(
            df_plot,
            x='month_year',
            y='value',
            color='year',
            title=title,
            labels={'month_year': 'Monat / Jahr', 'value': y_label, 'year': 'Jahr'},
            # HIER IST DIE ÄNDERUNG:
            color_discrete_sequence=["#81c784", "#4caf50", "#388e3c"],
            text_auto='.2s'
        )

        fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': df_plot['month_year']})
        st.plotly_chart(fig, use_container_width=True)


    # 4. Logik zur Anzeige der Plots
    # =================================================================
    if selected_metric == 'Warenwert':
        plot_metric('revenue')
    elif selected_metric == 'Mitglieder':
        plot_metric('customers')
    elif selected_metric == 'Nicht bestellte Produkte':
        plot_metric('not_ordered')
    elif selected_metric == 'Alle':
        plot_metric('revenue')
        plot_metric('customers')
        plot_metric('not_ordered')

######################################################################################################################


def build_order_blocks_forecast(order_items, block_size=3):


    # (Code ist identisch zum vorherigen Modul, aber mit anderem Namen zur Sicherheit)
    df = order_items.copy()
    df['date_created'] = pd.to_datetime(df['date_created'], errors='coerce')
    df = df.dropna(subset=['date_created']).set_index('date_created')

    weekly = df.resample('W-Mon').size().reset_index(name='dummy')

    grouped_blocks = []
    current_block = []
    for _, row in weekly.iterrows():
        if row['dummy'] > 0:
            current_block.append(row)
            if len(current_block) == block_size:
                start_date = current_block[0]['date_created']
                end_date = current_block[-1]['date_created']
                mid_date = start_date + (end_date - start_date) / 2
                grouped_blocks.append({'start_date': start_date, 'end_date': end_date, 'date_created': mid_date})
                current_block = []
        else:
            if current_block:
                start_date = current_block[0]['date_created']
                end_date = current_block[-1]['date_created']
                mid_date = start_date + (end_date - start_date) / 2
                grouped_blocks.append({'start_date': start_date, 'end_date': end_date, 'date_created': mid_date})
                current_block = []

    if current_block:
        start_date = current_block[0]['date_created']
        end_date = current_block[-1]['date_created']
        mid_date = start_date + (end_date - start_date) / 2
        grouped_blocks.append({'start_date': start_date, 'end_date': end_date, 'date_created': mid_date})

    df_blocks = pd.DataFrame(grouped_blocks)
    if not df_blocks.empty:
        df_blocks['year'] = df_blocks['date_created'].dt.year
        df_blocks['month_year'] = df_blocks['date_created'].dt.strftime('%b %Y')
    return df_blocks


def modul_jahresprognose(dfs):
    st.subheader("📊 Jahresprognose des Warenwerts")


    # 1. Daten-Prüfung und -Vorbereitung
    # =================================================================
    if 'order_items' not in dfs or 'products' not in dfs:
        st.warning("Benötigte Daten ('order_items', 'products') nicht gefunden.")
        return

    order_items = dfs['order_items'].copy()
    products = dfs['products'].copy()

    # 2. Historische Blockdaten berechnen
    # =================================================================
    block_template = build_order_blocks_forecast(order_items, block_size=3)

    if block_template is None or block_template.empty:
        st.warning("Keine Bestellblöcke für die Prognose gefunden.")
        return

    # Warenwert pro Woche
    data = order_items.merge(products[['id', 'price_per_unit', 'tax_rate']], left_on='product_id_id', right_on='id')
    data['quantity'] = pd.to_numeric(data['quantity'], errors='coerce').fillna(0)
    data['gross_value'] = data['quantity'] * data['price_per_unit'] * (1 + data['tax_rate'])
    weekly_data = data.set_index('date_created').resample('W-Mon')['gross_value'].sum().reset_index(name='value').set_index(
        'date_created')

    # Warenwert pro Block
    block_values = []
    for _, block in block_template.iterrows():
        mask = (weekly_data.index >= block['start_date']) & (weekly_data.index <= block['end_date'])
        block_sum = weekly_data.loc[mask, 'value'].sum() if mask.any() else 0
        block_values.append({'date_created': block['date_created'], 'year': block['year'], 'value': block_sum})
    df_revenue_blocks = pd.DataFrame(block_values)

    # 3. Jahreswerte hochrechnen ("Completeness Factor")
    # =================================================================
    max_blocks_per_year = df_revenue_blocks.groupby('year').size().max()

    yearly_data = []
    for year, group in df_revenue_blocks.groupby('year'):
        blocks_present = len(group)
        completeness = blocks_present / max_blocks_per_year if max_blocks_per_year > 0 else 0
        total_value = group['value'].sum()
        adjusted_value = total_value / completeness if completeness > 0 else 0
        yearly_data.append({'year': year, 'completeness': completeness, 'adjusted_value': adjusted_value})
    df_yearly = pd.DataFrame(yearly_data).dropna().sort_values('year')

    # 4. Lineares Regressionsmodell trainieren
    # =================================================================
    df_yearly['year_index'] = np.arange(len(df_yearly))
    X = df_yearly[['year_index']]
    y = df_yearly['adjusted_value']

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    # 5. Prognose für das nächste Jahr
    # =================================================================
    last_year = df_yearly['year'].max()
    last_year_index = df_yearly['year_index'].max()

    future_index = pd.DataFrame({'year_index': [last_year_index + 1]})
    predicted_next_year_value = model.predict(future_index)[0]
    forecast_year = int(last_year + 1)

    # 6. Ergebnisse und Visualisierung in Streamlit
    # =================================================================
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Modellgüte (R²)", f"{r2:.3f}")
    with col2:
        st.metric(f"Prognose für {forecast_year}", f"{predicted_next_year_value:.2f} €")

    # Visualisierung
    fig = go.Figure()

    # Historische (hochgerechnete) Daten als Balken
    fig.add_trace(go.Bar(
        x=df_yearly['year'],
        y=df_yearly['adjusted_value'],
        name='Hochgerechneter Jahresumsatz',
        marker_color='#66bb6a'
    ))

    # Regressionsgerade
    y_pred_line = model.predict(X)
    fig.add_trace(go.Scatter(
        x=df_yearly['year'],
        y=y_pred_line,
        mode='lines',
        name='Linearer Trend',
        line=dict(color='#388e3c', dash='dash')
    ))

    # Prognosepunkt
    fig.add_trace(go.Scatter(
        x=[forecast_year],
        y=[predicted_next_year_value],
        mode='markers',
        name=f'Prognose {forecast_year}',
        marker=dict(color='orange', size=12, symbol='star')
    ))

    fig.update_layout(
        title='Jahresumsatz-Prognose (basierend auf linearem Trend)',
        xaxis_title='Jahr',
        yaxis_title='Hochgerechneter Warenwert (€)'
    )
    st.plotly_chart(fig, use_container_width=True)

########################################################################################################################

def modul_mitglieder_zeit(dfs):
    st.subheader("📈 Mitgliederentwicklung über die Zeit")

    # 1. Daten-Prüfung und -Vorbereitung
    if 'alle_mitglieder' not in dfs or dfs['alle_mitglieder'].empty:
        st.warning("Benötigte Daten ('alle_mitglieder') nicht gefunden oder leer.")
        return

    df_members = dfs['alle_mitglieder'].copy()

    # Sonderfall "Gründungsmitglied" -> auf erstes tatsächliches Datum setzen
    try:
        min_date = pd.to_datetime(
            df_members.loc[~df_members['Antrag'].str.contains("Gründungsmitglied", na=False), 'Antrag'],
            errors='coerce'
        ).min()

        df_members['Antrag_clean'] = df_members['Antrag'].apply(
            lambda x: min_date if isinstance(x, str) and "Gründungsmitglied" in x else x
        )
        df_members['Antrag_clean'] = pd.to_datetime(df_members['Antrag_clean'], errors='coerce')
    except Exception as e:
        st.error(f"Fehler bei der Datumsbereinigung: {e}")
        return

    # 2. Fehlerhafte Werte entfernen und filtern
    df_members = df_members.dropna(subset=['Antrag_clean'])
    df_members = df_members[df_members['Antrag_clean'] >= pd.Timestamp('2021-01-01')]

    if df_members.empty:
        st.warning("Nach der Datenbereinigung sind keine Mitgliederdaten mehr vorhanden.")
        return

    # 3. Gruppierung nach Woche und Vorbereitung für Plot
    df_members = df_members.set_index('Antrag_clean')
    weekly_counts = df_members.resample('W-Mon').size().reset_index(name='count')
    weekly_counts['year'] = weekly_counts['Antrag_clean'].dt.year
    weekly_counts['month_year'] = weekly_counts['Antrag_clean'].dt.strftime('%b %Y')

    # 4. Plot erstellen
    fig, ax = plt.subplots(figsize=(15, 6))

    jahre = sorted(weekly_counts['year'].unique())
    farben = sns.color_palette("Greens", n_colors=len(jahre))
    jahr_farben = {jahr: farben[i] for i, jahr in enumerate(jahre)}

    sns.barplot(
        data=weekly_counts,
        x='Antrag_clean',
        y='count',
        hue='year',
        dodge=False,
        palette=jahr_farben,
        ax=ax
    )

    ax.set_title("Anzahl neuer Mitglieder pro Woche (ab 2021)", fontsize=16)
    ax.set_xlabel("Monat / Jahr", fontsize=12)
    ax.set_ylabel("Anzahl neuer Mitglieder", fontsize=12)

    # X-Achse lesbarer machen
    tick_indices = range(0, len(weekly_counts), 4)
    ax.set_xticks(ticks=tick_indices)
    ax.set_xticklabels(weekly_counts['month_year'].iloc[::4], rotation=45, ha="right")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # 5. Gesamtanzahl pro Jahr in einem Expander anzeigen
    with st.expander("Gesamtanzahl neuer Mitglieder pro Jahr anzeigen"):
        year_totals = weekly_counts.groupby('year')['count'].sum().reset_index()
        year_totals = year_totals.rename(columns={'year': 'Jahr', 'count': 'Anzahl neuer Mitglieder'})
        st.dataframe(year_totals, use_container_width=True, hide_index=True)


########################################################################################################################

# In deine app.py einfügen oder als separate Datei importieren
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point
from streamlit_folium import st_folium


def modul_mitglieder_karte(dfs):
    st.subheader("🗺️ Geografische Verteilung der Mitglieder im Zeitverlauf")

    # 1. Daten-Prüfung und -Vorbereitung
    if 'alle_mitglieder_geo' not in dfs or dfs['alle_mitglieder_geo'].empty:
        st.warning("Benötigte Daten ('alle_mitglieder_geo') nicht gefunden oder leer.")
        return

    df_geo = dfs['alle_mitglieder_geo'].dropna(subset=['latitude', 'longitude']).copy()
    df_geo['Antrag'] = pd.to_datetime(df_geo['Antrag'], errors='coerce')
    df_geo = df_geo.dropna(subset=['Antrag'])
    df_geo = df_geo[df_geo['Antrag'] >= pd.Timestamp('2021-01-01')]

    if df_geo.empty:
        st.warning("Nach der Datenbereinigung sind keine Geodaten mehr vorhanden.")
        return

    # 2. GeoJSON für Stadtteile laden
    try:
        gebiete = gpd.read_file("stadtteile-statistische-bezirke-muenster.geojson").to_crs(epsg=25832)
        if "name" not in gebiete.columns:
            gebiete['name'] = gebiete.index.astype(str)
    except FileNotFoundError:
        st.error(
            "Fehler: Die Datei 'stadtteile-statistische-bezirke-muenster.geojson' wurde nicht gefunden. Bitte stelle sicher, dass sie im Hauptverzeichnis liegt.")
        return

    gdf_points_full = gpd.GeoDataFrame(
        df_geo,
        geometry=[Point(xy) for xy in zip(df_geo.longitude, df_geo.latitude)],
        crs="EPSG:4326"
    ).to_crs(epsg=25832)

    # 3. Interaktiven Zeitstrahl (Slider) erstellen
    min_date = pd.Timestamp('2022-06-01')
    max_date = df_geo['Antrag'].max().replace(day=1)
    date_range = pd.date_range(min_date, max_date, freq='MS')

    selected_date = st.select_slider(
        "Wähle einen Monat aus, um die Mitgliederverteilung bis zu diesem Zeitpunkt zu sehen:",
        options=date_range,
        value=max_date,
        format_func=lambda date: date.strftime('%B %Y'),
        key="time_slider_map"
    )

    # 4. Karte basierend auf dem ausgewählten Datum erstellen
    gdf_points = gdf_points_full[gdf_points_full['Antrag'] <= selected_date]
    joined = gpd.sjoin(gdf_points, gebiete, how="left", predicate='within')
    joined['gebiet'] = joined['name'].fillna("Außerhalb Münster")
    gebiet_counts = joined.groupby('gebiet').size().reset_index(name='count')
    outside_count = int(gebiet_counts.loc[gebiet_counts['gebiet'] == "Außerhalb Münster", "count"].sum())
    gebiet_counts = gebiet_counts[gebiet_counts['gebiet'] != "Außerhalb Münster"]

    gebiete_counts = gebiete.merge(gebiet_counts, left_on='name', right_on='gebiet', how='left')
    gebiete_counts['count'] = gebiete_counts['count'].fillna(0)
    gebiete_counts['gebiet'] = gebiete_counts['gebiet'].fillna(gebiete_counts['name']).astype(str)
    gebiete_counts = gebiete_counts.to_crs(epsg=4326)

    # =========================================================================
    # HIER IST DIE LÖSUNG: Reduziere den DataFrame auf die benötigten Spalten
    # Dadurch wird die problematische Timestamp-Spalte entfernt.
    # =========================================================================
    gebiete_counts_for_map = gebiete_counts[['gebiet', 'count', 'geometry']]

    # 5. Folium-Karte erstellen und anzeigen
    m = folium.Map(location=[51.96236, 7.62571], zoom_start=12, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=gebiete_counts_for_map,  # <- HIER den bereinigten DataFrame verwenden
        data=gebiete_counts_for_map,  # <- HIER ebenfalls
        columns=['gebiet', 'count'],
        key_on='feature.properties.gebiet',
        bins=list(range(0, 21, 2)),
        fill_color='Greens',
        nan_fill_color='white',
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name=f"Mitglieder pro Gebiet bis {selected_date.strftime('%B %Y')} (zzgl. {outside_count} außerhalb)"
    ).add_to(m)

    # Feste Standorte
    standorte = [
        {"name": "Alter Standort", "coords": (51.96590, 7.62348)},
        {"name": "Neuer Standort (seit 2024)", "coords": (51.96198, 7.64271)}
    ]
    for s in standorte:
        folium.Marker(
            location=s["coords"],
            popup=folium.Popup(s["name"], max_width=250),
            icon=folium.Icon(color="blue", icon="shopping-cart")
        ).add_to(m)

    st_folium(m, width=725, height=500)

########################################################################################################################

# ========== Globale Plot-Einstellungen ==========
sns.set_theme(style="whitegrid", palette="Greens", font_scale=1.1)
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# ========== Seitentitel & Sidebar ==========
st.set_page_config(page_title="Namiko Data Analysis", layout="wide")
st.title("📊 Namiko Münster – Analyse-Tool")

analysis_options = [
    "Mitgliederübersicht",
    "Geografische Auswertung",
    "Lieferantenanalyse",
    "Produktkategorien",
    "Solidaritätssteuer",
    "Bestellhäufigkeit",
    "Top / Flop Produkte",
    "Bestellungen nach Wochentagen",
    "Warenwert / Mitglieder / Nicht bestellt",
    "Jahresprognose",
    "Mitgliederanalyse nach Zeit",
    "Mitgliederkarte mit Zeitstrahl"
]
selected_analysis = st.sidebar.selectbox("Analyse auswählen:", analysis_options)

# ========== Feste Dateien laden ==========
geojson_path = r"C:\Users\Benjamin\Documents\Data-Science Hausarbeit\stadtteile-statistische-bezirke-muenster.geojson"
if os.path.exists(geojson_path):
    gebiete = gpd.read_file(geojson_path)
else:
    st.error("GeoJSON-Datei nicht gefunden. Bitte Pfad prüfen.")
    st.stop()

# ========== Upload data.7z ==========
uploaded_archive = st.file_uploader("Bitte die Datei data.7z hochladen", type=["7z"])
dfs = {}

if uploaded_archive is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "data.7z")
        with open(archive_path, "wb") as f:
            f.write(uploaded_archive.read())

        try:
            # Für Passwortschutz ggf. Passwortfeld einbauen:
            password = st.text_input("Passwort für Archiv:", type="password")
            #password = None
            with py7zr.SevenZipFile(archive_path, mode='r', password=password) as archive:
                archive.extractall(path=tmpdir)

            file_map = {
                'customers': 'data/customers.csv',
                'suppliers': 'data/suppliers.csv',
                'product_categories': 'data/product_categories.csv',
                'balance_sheet': 'data/balance_sheet.csv',
                'products': 'data/products.csv',
                'orders': 'data/orders.csv',
                'invoices': 'data/invoices.csv',
                'mitglieder': 'data/mitglieder_verzeichnis.xlsx',
                'mitglieder_alt': 'data/mitglieder_verzeichnis_alt.xlsx',
                'order_items': 'data/order_items.json'
            }

            for key, filename in file_map.items():
                full_path = os.path.join(tmpdir, filename)
                if os.path.exists(full_path):
                    if filename.endswith(".csv"):
                        df = pd.read_csv(full_path)
                    elif filename.endswith(".xlsx"):
                        df = pd.read_excel(full_path)
                    elif filename.endswith(".json"):
                        with open(full_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        df = pd.json_normalize(data, sep="_")
                    else:
                        continue
                    dfs[key] = df

            st.success("✅ Daten wurden erfolgreich geladen und verarbeitet.")

        except Exception as e:
            st.error(f"Fehler beim Entpacken/Laden. Ggf. Eingabe von Passwort notwendig.")

# ========== Automatische Vorverarbeitung ==========
if dfs:
    # Mitglieder zusammenführen
    spalten_bis_ort = ['Anrede', 'Vorname', 'Nachname', 'Antrag', 'E-Mail', 'Tel.',
                       'Signal', 'Mail/Signal', 'Straße + Hausnummer', 'PLZ + Ort']
    mitglieder_neu = dfs['mitglieder'][spalten_bis_ort].copy()
    mitglieder_alt = dfs['mitglieder_alt'][spalten_bis_ort].copy()

    mitglieder_neu['Quelle'] = 'neu'
    mitglieder_alt['Quelle'] = 'alt'

    alle_mitglieder = pd.concat([mitglieder_neu, mitglieder_alt], ignore_index=True)
    alle_mitglieder.sort_values(by='Quelle', inplace=True)
    alle_mitglieder.drop_duplicates(subset=['Vorname', 'Nachname'], keep='first', inplace=True)
    alle_mitglieder.drop(columns=['Quelle'], inplace=True)

    kunden_ids = dfs['customers'][['first_name', 'last_name', 'id']].drop_duplicates(
        subset=['first_name', 'last_name'])
    alle_mitglieder = alle_mitglieder.merge(
        kunden_ids,
        left_on=['Vorname', 'Nachname'],
        right_on=['first_name', 'last_name'],
        how='left'
    ).drop(columns=['first_name', 'last_name'])

    dfs['alle_mitglieder'] = alle_mitglieder

    # Datentyp-Anpassungen & NaN-Analyse
    # (vereinfachte Version)
    for name, df in dfs.items():
        for col in df.columns:
            if "date" in col.lower():
                dfs[name][col] = pd.to_datetime(dfs[name][col], errors='coerce')

    st.info("Vorverarbeitung abgeschlossen.")


# ========== Analyse-Module ==========
if dfs and selected_analysis:
    if selected_analysis == "Mitgliederübersicht":
        modul_mitgliederuebersicht(dfs)


    elif selected_analysis == "Geografische Auswertung":

        modul_geografische_auswertung(dfs)

    elif selected_analysis == "Lieferantenanalyse":

        modul_lieferantenanalyse(dfs)

    elif selected_analysis == "Produktkategorien":

        modul_produktkategorien(dfs)

    elif selected_analysis == "Solidaritätssteuer":

        modul_solidaritaetssteuer(dfs)

    elif selected_analysis == "Bestellhäufigkeit":

        modul_bestellhaeufigkeit(dfs)

    elif selected_analysis == "Top / Flop Produkte":

        modul_top_flop(dfs)

    elif selected_analysis == "Bestellungen nach Wochentagen":

        modul_wochentage(dfs)

    elif selected_analysis == "Warenwert / Mitglieder / Nicht bestellt":
        modul_bestellbloecke(dfs)

    elif selected_analysis == "Jahresprognose":
        modul_jahresprognose(dfs)

    elif selected_analysis == "Mitgliederanalyse nach Zeit":  # NEU
        modul_mitglieder_zeit(dfs)


    elif selected_analysis == "Mitgliederkarte mit Zeitstrahl":

        # HIER PRÜFEN WIR, OB DIE GEO-DATEN IM SESSION STATE SIND

        if 'geo_df' in st.session_state:

            dfs['alle_mitglieder_geo'] = st.session_state['geo_df']

            modul_mitglieder_karte(dfs)

        else:

            st.warning("Bitte lade zuerst die 'Alle Mitglieder Geo'-Datei hoch, um diese Analyse durchzuführen.")


