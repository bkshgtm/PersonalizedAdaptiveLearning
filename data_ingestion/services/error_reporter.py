import csv
import io
from typing import List, Dict
from django.core.files.base import ContentFile
from data_ingestion.models import DataUpload

class ErrorReporter:
    """
    Handles reporting and exporting of CSV import errors.
    """
    
    def __init__(self, data_upload: DataUpload):
        self.data_upload = data_upload
        
    def generate_error_report(self, failed_rows: List[Dict]) -> ContentFile:
        """
        Generate a CSV file containing all failed rows with error details.
        
        Args:
            failed_rows: List of dictionaries containing failed row data
            
        Returns:
            Django ContentFile with the error report
        """
        if not failed_rows:
            return None
            
        # Create CSV output
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        headers = list(failed_rows[0]['row_data'].keys()) + ['error_message']
        writer.writerow(headers)
        
        # Write failed rows
        for row in failed_rows:
            row_data = list(row['row_data'].values()) + [row['error']]
            writer.writerow(row_data)
        
        # Create and return file
        content = output.getvalue()
        output.close()
        return ContentFile(content.encode('utf-8'), name='error_report.csv')
        
    def save_error_report(self, failed_rows: List[Dict]) -> None:
        """
        Save error report to the data upload record.
        """
        error_file = self.generate_error_report(failed_rows)
        if error_file:
            self.data_upload.error_file.save('error_report.csv', error_file)
            self.data_upload.save()
