import gradio as gr
from deal_agent_framework import DealAgentFramework
from agents.deals import Opportunity, Deal
import threading

class App:
    instance_count = 0
    _framework_lock = threading.Lock()

    def __init__(self):
        App.instance_count += 1  
        self.agent_framework = None

    def run(self):
        with gr.Blocks(title="The Price is Right", fill_width=True) as ui:
        
            def table_for(opps):
                return [[opp.deal.product_description, f"${opp.deal.price:.2f}", f"${opp.estimate:.2f}", f"${opp.discount:.2f}", opp.deal.url] for opp in opps]
        
            def start():
                print(f"In start, App instance #{App.instance_count}")
                with App._framework_lock:
                    if self.agent_framework is None:
                        print(f"In start, App._framework_lock acquired")
                        self.agent_framework = DealAgentFramework()
                        # redundant as self.agent_framework.init_agents_as_needed() called in run
                        # self.agent_framework.init_agents_as_needed()
                print(f"In start, App._framework_lock released")
                opportunities = self.agent_framework.get_memory() 
                table = table_for(opportunities)
                return table
            # go is called every 60 seconds by the Timer component only
            def go():
                '''
                Run agent framework to update the list of opporunities with new 
                a new fully processed one per call
                '''
                with App._framework_lock:
                    print(f"In go, App._framework_lock acquired")
                    if self.agent_framework is None:
                        print(f"In go, App._framework_lock return")
                        return
                print(f"In go, App._framework_lock released")
                self.agent_framework.run()   
                new_opportunities = self.agent_framework.get_memory()
                table = table_for(new_opportunities)
                return table
        
            def do_select(selected_index: gr.SelectData):
                with App._framework_lock:
                    print(f"In do_select, App._framework_lock acquired")
                    if self.agent_framework is None:
                        print(f"In do_select, App._framework_lock return")
                        return

                print(f"In do_select, App._framework_lock released")    
                opportunities = self.agent_framework.get_memory()
                row = selected_index.index[0]
                if row < 0 or row >= len(opportunities):
                    return
                opportunity = opportunities[row]
                self.agent_framework.planner.messenger.alert(opportunity)
        
            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:24px">"The Price is Right" - Deal Hunting Agentic AI</div>')
            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:14px">Autonomous agent framework that finds online deals, collaborating with a proprietary fine-tuned LLM deployed on Modal, and a RAG pipeline with a frontier model and Chroma.</div>')
            with gr.Row():
                gr.Markdown('<div style="text-align: center;font-size:14px">Deals surfaced so far:</div>')
            with gr.Row():
                opportunities_dataframe = gr.Dataframe(
                    headers=["Description", "Price", "Estimate", "Discount", "URL"],
                    wrap=True,
                    column_widths=[4, 1, 1, 1, 2],
                    row_count=10,
                    col_count=5,
                    max_height=400,
                )
        
            ui.load(start, inputs=[], outputs=[opportunities_dataframe])
            
            print(f"In run, before timer, App instance #{App.instance_count}")
            timer = gr.Timer(value=60)
            timer.tick(go, inputs=[], outputs=[opportunities_dataframe])
            print(f"In run, after timer, App instance #{App.instance_count}")

            opportunities_dataframe.select(do_select)
        
        ui.launch(share=False, inbrowser=True)

if __name__=="__main__":
    App().run()
    