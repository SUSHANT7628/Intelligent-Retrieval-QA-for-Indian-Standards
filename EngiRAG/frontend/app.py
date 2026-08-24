import streamlit as st
from src.rag_pipeline import EngiRAG
from src.query.rewriter import QueryRewriter

st.set_page_config(page_title='EngiRAG', page_icon='📐', layout='wide')
st.title('📐 EngiRAG')
st.caption('Engineering document intelligence with hybrid retrieval, reranking, query rewriting and cited answers')

@st.cache_resource
def load_engine(): return EngiRAG()
@st.cache_resource
def load_rewriter(): return QueryRewriter()

engine=load_engine(); rewriter=load_rewriter()
question=st.text_area('Ask an engineering question', placeholder='Compare the concrete cover requirements described in the available documents.')
k=st.slider('Evidence chunks',3,10,5)

if st.button('Search & Answer', type='primary') and question.strip():
    queries=rewriter.rewrite(question)
    st.write('**Retrieval queries:**', queries)
    with st.spinner('Retrieving and reranking evidence...'):
        result=engine.answer(question,k=k)
    st.subheader('Answer')
    st.write(result['answer'])
    if result.get('citation_check'):
        st.json(result['citation_check'])
    st.subheader('Retrieved evidence')
    for i,ctx in enumerate(result['contexts'],1):
        meta=ctx.get('metadata',{})
        with st.expander(f"[{i}] {meta.get('source','unknown')} — page {meta.get('page','?')}"):
            st.write(ctx['text'])
