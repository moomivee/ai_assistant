"""Real SDK + native callbacks, local OTLP collector; does not require cloud keys."""

import subprocess
import sys


def test_native_langgraph_and_tools_share_exported_trace(tmp_path):
    script = r"""
import gzip,json,os,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
batches=[]
class Collector(BaseHTTPRequestHandler):
    def do_POST(self):
        body=self.rfile.read(int(self.headers['Content-Length']))
        if self.headers.get('Content-Encoding')=='gzip': body=gzip.decompress(body)
        if self.path.endswith('/v1/traces'):
            msg=ExportTraceServiceRequest();msg.ParseFromString(body);batches.append(msg)
        self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(b'{}')
    def log_message(self,*args):pass
server=ThreadingHTTPServer(('127.0.0.1',0),Collector)
threading.Thread(target=server.serve_forever,daemon=True).start()
os.environ.update(LANGFUSE_ENABLED='true',LANGFUSE_PUBLIC_KEY='pk-lf-test-export',LANGFUSE_SECRET_KEY='sk-lf-test-export',
 LANGFUSE_BASE_URL=f'http://127.0.0.1:{server.server_port}',TRACE_CONTENT='false',OTEL_BSP_SCHEDULE_DELAY='100')
from src.config import settings
from src.agent import ask
from src.models.schemas import Plan
from src.integrations.telemetry import flush,langfuse_client
from src.tools import run_tool
class Client:
 def generate(self,name,payload,schema):
  return Plan(intent='forecast',rag_query='oil',fresh=False,report_only=False,wants_context=False,news=False,quotes=[],
    benchmark='Brent',months=3,discount=None,cut=None,sensitivity=None,parameters=[],clarify='')
def execute(registry,name,**args):
 if name=='price_forecast':return run_tool(registry,name,**args)
 return [],None,{'tool':name,'engine':'fixture','status':'ok','ms':0}
result=ask('PRIVATE-MARKER Brent на 3 месяца',client=Client(),execute=execute)
flush()
client=langfuse_client();client.shutdown()
server.shutdown()
spans=[s for b in batches for r in b.resource_spans for scope in r.scope_spans for s in scope.spans]
names={s.name for s in spans}
assert {'agent.request','LangGraph','price_forecast','workflow.plan'} <= names,names
assert {s.trace_id.hex() for s in spans}=={result['trace_id']}
root=next(s for s in spans if s.name=='agent.request')
graph=next(s for s in spans if s.name=='LangGraph')
assert graph.parent_span_id==root.span_id
assert 'PRIVATE-MARKER' not in str(batches)
print(json.dumps({'spans':len(spans),'names':sorted(names),'status':result['status']}))
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=45)
    assert r.returncode == 0, r.stdout + "\n" + r.stderr
