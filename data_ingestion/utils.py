from django.utils.html import format_html

def format_validation_metadata(metadata):
    """Formats validation metadata for display in admin/templates"""
    if not metadata:
        return "No detailed validation available"
    
    def format_value(value, indent=0):
        if isinstance(value, dict):
            html = []
            for k, v in value.items():
                html.append(f"{'&nbsp;' * indent * 4}&bull; {k}: {format_value(v, indent+1)}")
            return "<br>".join(html)
        elif isinstance(value, list):
            return ", ".join(str(x) for x in value)
        return str(value)
    
    return format_html(format_value(metadata))
