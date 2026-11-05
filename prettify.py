# # Example Usage:
# print_card(
#     title="Data Processing Complete", 
#     content="Pipeline executed successfully. 1,240 rows processed in 0.42 seconds.",
#     border_color="#28a745" # Green accent
# )

from IPython.display import display, HTML

def print_card(title, content, border_color="#007acc"):
    card_html = f"""
    <div style="
        border-left: 5px solid {border_color};
        background-color: #f8f9fa;
        padding: 15px 20px;
        margin: 10px 0;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    ">
        <h4 style="margin: 0 0 8px 0; color: #333; font-weight: 600;">{title}</h4>
        <p style="margin: 0; color: #555; line-height: 1.5; font-size: 14px;">{content}</p>
    </div>
    """
    display(HTML(card_html))
