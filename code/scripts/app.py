"""CancelIQ: grounded booking analytics with optional Gemini narration."""
import json, os, re, urllib.request, urllib.error
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = pd.read_csv(ROOT / 'bookings_scored.csv', low_memory=False)
META = json.loads((ROOT / 'best_model_metadata.json').read_text())
DIMENSIONS = {'hotel':'hotel','hotel type':'hotel','الفندق':'hotel','market':'market_segment','segment':'market_segment','السوق':'market_segment','القناة':'distribution_channel','deposit':'deposit_type','وديعة':'deposit_type','risk':'risk_level','مخاطر':'risk_level','month':'arrival_month','شهر':'arrival_month','year':'arrival_year','سنة':'arrival_year','country':'country','دولة':'country','lead time':'lead_time_category','مدة':'lead_time_category'}
# The scored export has no distribution_channel: only expose present columns.
DIMENSIONS = {k:v for k,v in DIMENSIONS.items() if v in DATA.columns}
NUMBERS = {'bookings':'عدد الحجوزات','cancellations':'الإلغاءات','cancellation_rate':'نسبة الإلغاء','avg_risk':'متوسط المخاطرة المتوقعة','booked_revenue':'قيمة الحجوزات المقدّرة','lost_revenue':'قيمة الحجوزات الملغاة المقدّرة','expected_lost':'الخسارة المتوقعة بالنموذج'}

def compute(group=None, filters=None):
    frame=DATA
    for field,value in (filters or {}).items():
        if field not in DIMENSIONS.values(): raise ValueError('عمود غير مسموح')
        frame=frame[frame[field].astype(str).str.casefold()==str(value).casefold()]
    if group and group not in DIMENSIONS.values(): raise ValueError('تجميع غير مسموح')
    def row(d):
        return {'bookings':len(d),'cancellations':int(d.is_canceled.sum()),'cancellation_rate':round(float(d.is_canceled.mean()),4) if len(d) else None,'avg_risk':round(float(d.risk_prob.mean()),4) if len(d) else None,'booked_revenue':round(float(d.estimated_revenue.sum()),2),'lost_revenue':round(float(d.loc[d.is_canceled.eq(1),'estimated_revenue'].sum()),2),'expected_lost':round(float(d.expected_lost_revenue.sum()),2)}
    if group:
        rows=[{'group':str(value),**row(part)} for value,part in frame.groupby(group,dropna=False)]
        return {'source':'bookings_scored.csv','group_by':group,'filters':filters or {},'rows':sorted(rows,key=lambda x:x['bookings'],reverse=True)[:20]}
    return {'source':'bookings_scored.csv','filters':filters or {},'summary':row(frame)}

def local_plan(question):
    q=question.casefold()
    group=next((v for k,v in sorted(DIMENSIONS.items(),key=lambda x:-len(x[0])) if k in q),None)
    if 'مقارن' in q and group is None: group='hotel'
    return {'group_by':group,'filters':{}}

def gemini(prompt):
    key=os.environ.get('GEMINI_API_KEY')
    if not key: return None
    model=os.environ.get('GEMINI_MODEL','gemini-2.5-flash')
    url=f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
    payload={'contents':[{'parts':[{'text':prompt}]}],'generationConfig':{'temperature':0}}
    req=urllib.request.Request(url,json.dumps(payload).encode(),headers={'Content-Type':'application/json','x-goog-api-key':key})
    try:
        with urllib.request.urlopen(req,timeout=25) as response: data=json.load(response)
        return ''.join(p.get('text','') for p in data['candidates'][0]['content']['parts']).strip()
    except (urllib.error.URLError, KeyError, IndexError, TimeoutError) as exc:
        return None

def answer(question):
    if not question or len(question)>1000: raise ValueError('اكتب سؤالًا من 1 إلى 1000 حرف')
    allowed={k:sorted(DATA[v].dropna().astype(str).unique().tolist())[:80] for k,v in DIMENSIONS.items() if k==v}
    plan=local_plan(question)
    planning=gemini('Return ONLY JSON {"group_by": null or a column, "filters": {column: exact value}}. Choose only from this allowlist. Never invent values. Question: '+question+'\nAllowlist: '+json.dumps(allowed,ensure_ascii=False))
    if planning:
        try:
            match=re.search(r'\{.*\}',planning,re.S)
            candidate=json.loads(match.group())
            col=candidate.get('group_by'); filters=candidate.get('filters',{})
            if (col is None or col in DIMENSIONS.values()) and isinstance(filters,dict) and all(k in allowed and str(v) in allowed[k] for k,v in filters.items()): plan={'group_by':col,'filters':filters}
        except (ValueError,AttributeError,TypeError): pass
    result=compute(plan['group_by'],plan['filters'])
    narrative=gemini('أجب بالمصري بإيجاز على السؤال باستخدام أرقام JSON فقط. النسب العشرية اضربها في 100؛ الأموال تقديرات ADR × الليالي وليست خسارة محاسبية مؤكدة. لو السؤال لا يمكن إجابته من JSON قل بوضوح. لا تدّعي بيانات حديثة أو أداء النموذج خارج test. السؤال: '+question+'\nالنتائج: '+json.dumps(result,ensure_ascii=False))
    if not narrative:
        if 'rows' in result: narrative='مقارنة حسب '+result['group_by']+':\n'+'\n'.join(f"{x['group']}: {x['bookings']:,} حجز، نسبة الإلغاء {x['cancellation_rate']:.1%}، مخاطرة متوقعة {x['avg_risk']:.1%}" for x in result['rows'][:10])
        else:
            x=result['summary']; narrative=f"عدد الحجوزات {x['bookings']:,}، الملغاة {x['cancellations']:,} ({x['cancellation_rate']:.1%})، متوسط المخاطرة {x['avg_risk']:.1%}، والقيمة المقدّرة للحجوزات الملغاة {x['lost_revenue']:,.2f}."
    return {'answer':narrative,'evidence':result,'llm_enabled':bool(os.environ.get('GEMINI_API_KEY') and planning)}

HTML='''<!doctype html><html lang="ar" dir="rtl"><meta charset="utf-8"><title>CancelIQ Assistant</title><style>body{font:18px system-ui;background:#f0f5f7;color:#152d3c;max-width:850px;margin:5vh auto;padding:24px}main{background:white;padding:32px;border-radius:20px;box-shadow:0 8px 30px #15334415}input{width:78%;padding:14px;border:1px solid #abc;border-radius:8px;font:inherit}button{background:#087d76;color:white;border:0;padding:15px;border-radius:8px;font:inherit;cursor:pointer}pre{white-space:pre-wrap;line-height:1.7;background:#eaf5f3;padding:20px;border-radius:12px}small{color:#667}</style><main><h1>CancelIQ · مساعد الحجوزات</h1><p>اسأل عن الإلغاءات، المخاطر، أو قارن الفنادق والشرائح والشهور.</p><input id="q" value="قارن نسبة الإلغاء حسب الفندق"><button onclick="ask()">اسأل</button><pre id="out">النتيجة ستظهر هنا</pre><small>المصدر: bookings_scored.csv · بيانات تاريخية 2015–2017. الأرقام المالية تقديرات.</small></main><script>async function ask(){let out=document.getElementById('out');out.textContent='جارٍ التحليل...';try{let r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:document.getElementById('q').value})});let d=await r.json();out.textContent=d.answer||d.error}catch(e){out.textContent=String(e)}}</script></html>'''
class Handler(BaseHTTPRequestHandler):
    def respond(self,code,payload,ctype='application/json; charset=utf-8'):
        raw=(payload if isinstance(payload,str) else json.dumps(payload,ensure_ascii=False)).encode(); self.send_response(code);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    def do_GET(self):
        if self.path=='/': self.respond(200,(ROOT / 'hotel_chatbot_updated.html').read_text(encoding='utf-8'),'text/html; charset=utf-8')
        elif self.path=='/api/health': self.respond(200,{'status':'ok','rows':len(DATA),'gemini_configured':bool(os.environ.get('GEMINI_API_KEY'))})
        elif self.path=='/api/metrics': self.respond(200,{'test_metrics':META['test_metrics'],'dataset':compute()})
        else:self.respond(404,{'error':'غير موجود'})
    def do_POST(self):
        if self.path!='/api/chat':return self.respond(404,{'error':'غير موجود'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if size>5000: raise ValueError('الطلب كبير')
            body=json.loads(self.rfile.read(size)); self.respond(200,answer(body.get('question','')))
        except (ValueError,TypeError) as exc:self.respond(400,{'error':str(exc)})
if __name__=='__main__':
    print('Open http://127.0.0.1:8000');ThreadingHTTPServer(('127.0.0.1',8000),Handler).serve_forever()
