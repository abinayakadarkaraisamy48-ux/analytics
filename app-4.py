import re, sqlite3
from pathlib import Path
from datetime import datetime, date
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title='SmartSpend AI', page_icon='💰', layout='wide')
DB = Path(__file__).with_name('smartspend.db')
CATS = ['Food','Travel','Shopping','Education','Entertainment','Bills','Recharge','Healthcare','Groceries','Others']
ESSENTIAL = {'Education','Bills','Recharge','Healthcare','Groceries'}
NONESSENTIAL = {'Food','Shopping','Entertainment'}
MODES = ['UPI','GPay','PhonePe','Paytm','Card','Cash','Net Banking','Other']
st.markdown('''<style>.block-container{padding-top:1.3rem}.hero{padding:22px;border-radius:16px;background:linear-gradient(120deg,#102a43,#176b87);color:white}.hero h1{margin:0}.hero p{margin:5px 0 0}</style><div class="hero"><h1>💰 SmartSpend AI</h1><p>AI-Based Student Expense Tracking, Spending Analysis and Prediction System</p></div>''', unsafe_allow_html=True)

@st.cache_data
def synthetic_data(n=9850, seed=42):
    rng=np.random.default_rng(seed)
    payees={'Food':['Swiggy','Zomato','Cafe','Restaurant'],'Travel':['Uber','Ola','Rapido','Metro'],'Shopping':['Amazon','Flipkart','Myntra'],'Education':['College Fees','Book Store','Course Fee','Stationery'],'Entertainment':['Netflix','Spotify','Cinema'],'Bills':['Electricity Bill','Internet Bill','Gas Bill'],'Recharge':['Jio Recharge','Airtel Recharge','Vi Recharge'],'Healthcare':['Pharmacy','Hospital','Clinic'],'Groceries':['Grocery Store','Supermarket','Milk Shop'],'Others':['UPI Transfer','Local Shop','Miscellaneous']}
    ranges={'Food':(40,900),'Travel':(15,700),'Shopping':(100,4000),'Education':(100,10000),'Entertainment':(80,1500),'Bills':(150,3500),'Recharge':(99,999),'Healthcare':(50,5000),'Groceries':(50,2500),'Others':(20,3000)}
    cats=rng.choice(CATS,n,p=np.array([.20,.12,.10,.10,.07,.08,.08,.05,.13,.07]))
    rows=[]
    for c in cats:
        lo,hi=ranges[c]
        rows.append([round(float(rng.uniform(lo,hi)),2),str(rng.choice(payees[c])),str(rng.choice(MODES[:5])),c])
    return pd.DataFrame(rows,columns=['amount','payee','payment_mode','category'])

@st.cache_resource
def train_model():
    data=synthetic_data(); X=data[['payee','payment_mode','amount']]; y=data.category
    xa,xb,ya,yb=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    prep=ColumnTransformer([('text',TfidfVectorizer(ngram_range=(1,2)),'payee'),('mode',OneHotEncoder(handle_unknown='ignore'),['payment_mode']),('amount','passthrough',['amount'])])
    model=Pipeline([('prep',prep),('rf',RandomForestClassifier(n_estimators=120,random_state=42,n_jobs=-1,class_weight='balanced'))])
    model.fit(xa,ya)
    return model,accuracy_score(yb,model.predict(xb)),data
MODEL, ACCURACY, SYNTH= train_model()

@st.cache_resource
def db_conn():
    c=sqlite3.connect(DB,check_same_thread=False)
    c.execute('''CREATE TABLE IF NOT EXISTS transactions(id INTEGER PRIMARY KEY AUTOINCREMENT,amount REAL,payee TEXT,txn_date TEXT,txn_time TEXT,payment_mode TEXT,status TEXT,category TEXT,essential_type TEXT,raw_sms TEXT,created_at TEXT)''')
    c.commit(); return c
CON=db_conn()
def get_data():
    d=pd.read_sql_query('SELECT * FROM transactions ORDER BY txn_date DESC,id DESC',CON)
    if not d.empty:
        d['txn_date']=pd.to_datetime(d.txn_date,errors='coerce'); d['amount']=pd.to_numeric(d.amount,errors='coerce').fillna(0)
    return d
def save_txn(amount,payee,dt,tm,mode,status,cat,sms):
    typ='Essential' if cat in ESSENTIAL else ('Non-essential' if cat in NONESSENTIAL else 'Other')
    CON.execute('INSERT INTO transactions(amount,payee,txn_date,txn_time,payment_mode,status,category,essential_type,raw_sms,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)',(float(amount),payee,dt.strftime('%Y-%m-%d'),tm,mode,status,cat,typ,sms,datetime.now().isoformat(timespec='seconds'))); CON.commit()
def parse_sms(s):
    r={'raw_sms':s,'amount':None,'payee':'','txn_date':None,'txn_time':'','payment_mode':'UPI','status':'Completed'}
    m=re.search(r'(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d{1,2})?)',s,re.I)
    if m:r['amount']=float(m.group(1).replace(',',''))
    for pat in [r'\bto\s+([A-Za-z][A-Za-z0-9 &._-]{1,40}?)(?=\s+(?:via|using|on|at|for|UPI|Ref|Txn|$))',r'\b(?:at|towards)\s+([A-Za-z][A-Za-z0-9 &._-]{1,35}?)(?=\s+(?:on|via|using|Ref|Txn|$))']:
        m=re.search(pat,s,re.I)
        if m:r['payee']=m.group(1).strip(' .,-');break
    m=re.search(r'\b(\d{2}[-/]\d{2}[-/]\d{4})\b',s)
    if m:
        try:r['txn_date']=datetime.strptime(m.group(1), '%d-%m-%Y' if '-' in m.group(1) else '%d/%m/%Y').date()
        except ValueError:pass
    if not r['txn_date']:
        m=re.search(r'\b(\d{4}-\d{2}-\d{2})\b',s)
        if m:
            try:r['txn_date']=datetime.strptime(m.group(1),'%Y-%m-%d').date()
            except ValueError:pass
    m=re.search(r'\b(\d{1,2}:\d{2}\s*(?:AM|PM)?)\b',s,re.I)
    if m:r['txn_time']=m.group(1).upper()
    lo=s.lower()
    r['status']='Failed' if any(w in lo for w in ['failed','declined','unsuccessful']) else ('Pending' if 'pending' in lo else 'Completed')
    r['payment_mode']='GPay' if 'gpay' in lo or 'google pay' in lo else ('UPI' if 'upi' in lo else 'Other')
    return r

def month_data(d):
    if d.empty:return pd.DataFrame(columns=['month','amount'])
    x=d.copy();x['month']=x.txn_date.dt.to_period('M').astype(str)
    return x.groupby('month',as_index=False).amount.sum().sort_values('month')

with st.sidebar:
    st.title('💰 SmartSpend AI')
    page=st.radio('Navigation',['Dashboard','Add Transaction','Transaction History','Expense Analysis','AI Predictions','Dataset & Model','About Project'])
    st.caption('Privacy-conscious student expense analysis')

df=get_data()
if page=='Dashboard':
    st.header('📊 Dashboard')
    if df.empty: st.info('No transactions yet. Choose Add Transaction to begin.')
    else:
        completed=df[df.status=='Completed']; total=completed.amount.sum(); month_total=completed[completed.txn_date.dt.to_period('M')==pd.Timestamp.today().to_period('M')].amount.sum()
        a,b,c,d=st.columns(4);a.metric('Total Expenses',f'₹{total:,.2f}');b.metric('Transactions',len(df));c.metric('This Month',f'₹{month_total:,.2f}');d.metric('Average Transaction',f'₹{completed.amount.mean() if len(completed) else 0:,.2f}')
        l,r=st.columns(2)
        with l:
            st.subheader('Spending by Category'); cs=completed.groupby('category').amount.sum().sort_values(ascending=False);fig,ax=plt.subplots();cs.plot(kind='bar',ax=ax);ax.set_ylabel('Amount (₹)');ax.tick_params(axis='x',rotation=35);fig.tight_layout();st.pyplot(fig);plt.close(fig)
        with r:
            st.subheader('Essential vs Non-essential');es=completed.groupby('essential_type').amount.sum();fig,ax=plt.subplots();ax.pie(es.values,labels=es.index,autopct='%1.1f%%',startangle=90);ax.axis('equal');st.pyplot(fig);plt.close(fig)
        st.subheader('Monthly Spending Trend');md=month_data(completed)
        if len(md):
            fig,ax=plt.subplots();ax.plot(md.month,md.amount,marker='o');ax.set_ylabel('Amount (₹)');ax.tick_params(axis='x',rotation=35);fig.tight_layout();st.pyplot(fig);plt.close(fig)
        st.subheader('Recent Transactions');st.dataframe(df[['txn_date','payee','amount','category','payment_mode','status']].head(10),use_container_width=True,hide_index=True)

elif page=='Add Transaction':
    st.header('➕ Add Transaction')
    sms=st.text_area('Paste UPI / GPay SMS',placeholder='Paid Rs.250 to Swiggy via UPI on 27-09-2026 at 08:30 PM')
    if st.button('Parse SMS',type='primary'):
        st.session_state.parsed=parse_sms(sms)
        p=st.session_state.parsed
        if p['amount'] is None:st.warning('Amount not detected; enter it manually below.')
        if not p['payee']:st.warning('Payee not detected; enter it manually below.')
        if not p['txn_date']:st.warning('Date not detected; enter it manually below.')
        st.success('SMS parsed. Verify all fields before saving.')
    p=st.session_state.get('parsed',{})
    amount0=float(p.get('amount') or 0); payee0=p.get('payee') or ''; date0=p.get('txn_date') or date.today()
    with st.form('add_form'):
        amount=st.number_input('Amount (₹)',min_value=0.0,value=amount0,step=10.0)
        payee=st.text_input('Payee / Merchant',value=payee0)
        txn_date=st.date_input('Transaction Date',value=date0)
        txn_time=st.text_input('Transaction Time',value=p.get('txn_time',''))
        mode=st.selectbox('Payment Mode',MODES,index=MODES.index(p['payment_mode']) if p.get('payment_mode') in MODES else 0)
        status=st.selectbox('Status',['Completed','Pending','Failed'],index=['Completed','Pending','Failed'].index(p.get('status','Completed')))
        suggested=MODEL.predict(pd.DataFrame([{'payee':payee or 'Unknown','payment_mode':mode,'amount':amount}]))[0] if amount>0 else 'Others'
        st.info(f'AI suggested category: **{suggested}**')
        category=st.selectbox('Confirm / Change Category',CATS,index=CATS.index(suggested))
        raw=st.text_area('Original SMS (optional)',value=p.get('raw_sms',sms))
        submit=st.form_submit_button('Save Transaction',type='primary')
        if submit:
            if amount<=0:st.error('Amount must be greater than zero.')
            elif not payee.strip():st.error('Enter a payee / merchant.')
            else:
                save_txn(amount,payee.strip(),txn_date,txn_time,mode,status,category,raw);st.session_state.pop('parsed',None);st.success('Transaction saved. Dashboard has been updated.');st.rerun()

elif page=='Transaction History':
    st.header('🧾 Transaction History')
    if df.empty:st.info('No transactions yet.')
    else:
        c1,c2=st.columns(2); cats=c1.multiselect('Filter categories',CATS); statuses=c2.multiselect('Filter status',['Completed','Pending','Failed'],default=['Completed','Pending','Failed']); f=df.copy()
        if cats:f=f[f.category.isin(cats)]
        if statuses:f=f[f.status.isin(statuses)]
        st.dataframe(f,use_container_width=True,hide_index=True);st.caption(f'{len(f)} records • Total ₹{f.amount.sum():,.2f}')
        st.download_button('Export CSV',f.to_csv(index=False).encode(),file_name='smartspend_transactions.csv',mime='text/csv')
        st.divider();del_id=st.number_input('Transaction ID to delete',min_value=1,step=1)
        if st.button('Delete transaction'):
            cur=CON.execute('DELETE FROM transactions WHERE id=?',(int(del_id),));CON.commit()
            if cur.rowcount:st.success('Transaction deleted.');st.rerun()
            else:st.error('ID not found.')

elif page=='Expense Analysis':
    st.header('🔎 Expense Analysis')
    if df.empty:st.info('Add transactions to view analysis.')
    else:
        done=df[df.status=='Completed'];summary=done.groupby('category').amount.agg(['count','sum','mean']).reset_index().rename(columns={'count':'Transactions','sum':'Total (₹)','mean':'Average (₹)'})
        st.subheader('Category summary');st.dataframe(summary.sort_values('Total (₹)',ascending=False),use_container_width=True,hide_index=True)
        st.subheader('Month-wise spending');md=month_data(done)
        if len(md):st.bar_chart(md.set_index('month').amount)
        st.subheader('Essential vs Non-essential');es=done.groupby('essential_type').amount.sum();fig,ax=plt.subplots();ax.pie(es.values,labels=es.index,autopct='%1.1f%%');ax.axis('equal');st.pyplot(fig);plt.close(fig)
        if len(summary):st.info(f"Highest spending category: **{summary.sort_values('Total (₹)',ascending=False).iloc[0]['category']}**")

elif page=='AI Predictions':
    st.header('🤖 AI Predictions & Savings Assistant');st.metric('Synthetic hold-out model accuracy',f'{ACCURACY:.1%}');st.caption('Accuracy is measured on synthetic test data and may not reflect real-world SMS transactions.')
    if df.empty:st.info('Add transactions to get personal spending insights.')
    else:
        done=df[df.status=='Completed'];md=month_data(done)
        if len(md)>=2:st.metric('Estimated next-month spending',f"₹{md.amount.tail(3).mean():,.2f}");st.caption('Simple baseline: average of up to the last 3 observed monthly totals; not a guaranteed forecast.')
        else:st.info('Add completed transactions across at least two months to see an estimate.')
        totals=done.groupby('essential_type').amount.sum();e=float(totals.get('Essential',0));ne=float(totals.get('Non-essential',0))
        if ne>e:st.warning(f'Non-essential spending (₹{ne:,.2f}) is higher than essential spending (₹{e:,.2f}). Review Food, Shopping and Entertainment.')
        else:st.success('Keep reviewing your budget and checking your weekly spending.')
        if not done.empty:
            top=done.groupby('category').amount.sum().idxmax();st.write(f'- Your highest recorded spending category is **{top}**.');st.write('- Set a monthly budget and review recurring subscriptions.');st.write('- Check essential and non-essential spending every week.')

elif page=='Dataset & Model':
    st.header('🧪 Dataset, EDA & Machine Learning');st.write('A synthetic dataset is generated for model training; no bank account is connected.')
    a,b,c=st.columns(3);a.metric('Synthetic rows',f'{len(SYNTH):,}');b.metric('Categories',len(CATS));c.metric('Hold-out accuracy',f'{ACCURACY:.1%}')
    st.dataframe(SYNTH.head(20),use_container_width=True,hide_index=True);st.download_button('Download synthetic training dataset',SYNTH.to_csv(index=False).encode(),file_name='synthetic_expenses.csv',mime='text/csv')
    st.subheader('Category distribution');st.bar_chart(SYNTH.category.value_counts().reindex(CATS))
    st.markdown('**Pipeline:** payee text TF-IDF + payment-mode one-hot encoding + amount feature → Random Forest classifier → stratified train/test split → hold-out accuracy. Always review and confirm predicted categories.')

else:
    st.header('ℹ️ About SmartSpend AI')
    st.markdown('''**Project title:** AI-Based Student Expense Tracking, Spending Analysis and Prediction System

**Objective:** Help students record expenses, analyze spending patterns and receive basic savings suggestions.

**Workflow:** Paste payment SMS → extract fields using a best-effort rule-based parser → predict category using a trained ML model → review fields → save transaction to SQLite → update dashboard.

**Categories:** Food, Travel, Shopping, Education, Entertainment, Bills, Recharge, Healthcare, Groceries and Others.

**Tech stack:** Python, Streamlit, Pandas, NumPy, Matplotlib, Scikit-learn and SQLite.

**Privacy / deployment note:** This starter stores data in a local SQLite file. Before using it as a shared online service, add authentication, user separation and secure persistent storage. SMS formats vary, so verify parsed values.''')
st.divider();st.caption('SmartSpend AI • Academic demonstration • Verify extracted details before saving.')
