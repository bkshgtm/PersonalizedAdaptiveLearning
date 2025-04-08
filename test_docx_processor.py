import sys

print("Testing DOCX processor (standalone)...")
print(f"Python path: {sys.path}")

try:
    import docx
    print(f"docx imported from: {docx.__file__}")
except ImportError as e:
    print(f"Failed to import docx: {str(e)}")
    sys.exit(1)

# Test basic docx functionality
try:
    from io import BytesIO
    doc = docx.Document()
    doc.add_paragraph("Test paragraph")
    buffer = BytesIO()
    doc.save(buffer)
    print("Basic docx operations successful")
except Exception as e:
    print(f"docx operation failed: {str(e)}")
    sys.exit(1)

print("All tests passed")
