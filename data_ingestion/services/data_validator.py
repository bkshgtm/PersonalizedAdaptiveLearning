import csv
import io
from typing import Dict, List, Tuple
import datetime

class DataValidator:
    """
    Service class to validate CSV files before processing.
    """
    
    REQUIRED_FIELDS = [
        'student_id', 'course_id', 'question_id', 'assessment_id',
        'correct', 'timestamp'
    ]
    
    NUMERIC_FIELDS = [
        'GPA', 'prior_knowledge_score', 'attendance_rate',
        'participation_score', 'score', 'time_taken',
        'total_time_spent_on_platform', 'average_time_per_session'
    ]
    
    DATE_FIELDS = [
        'last_login_date', 'assessment_date', 'timestamp'
    ]
    
    BOOLEAN_FIELDS = [
        'correct', 'proctored', 'resource_viewed_before_question'
    ]
    
    def __init__(self, file_obj):
        """
        Initialize with a file object.
        """
        self.file_obj = file_obj
        self.errors = []
        self.warnings = []
        
    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate the CSV file.
        
        Returns:
            Tuple containing:
            - Boolean indicating if validation passed
            - List of error messages
            - List of warning messages
        """
        try:
            # Reset file pointer to beginning
            self.file_obj.seek(0)
            
            # Read the CSV file
            content = self.file_obj.read().decode('utf-8')
            csv_reader = csv.DictReader(io.StringIO(content))
            
            # Check if there are headers
            if not csv_reader.fieldnames:
                self.errors.append("CSV file has no headers")
                return False, self.errors, self.warnings
            
            # Check for required fields
            missing_fields = [field for field in self.REQUIRED_FIELDS if field not in csv_reader.fieldnames]
            if missing_fields:
                self.errors.append(f"CSV file is missing required fields: {', '.join(missing_fields)}")
                return False, self.errors, self.warnings
            
            # Check data (sample the first 100 rows to avoid memory issues)
            row_count = 0
            for row in csv_reader:
                row_count += 1
                
                # Validate row data
                self._validate_row(row, row_count)
                
                # Just check a sample for large files
                if row_count >= 100:
                    break
            
            # Warn if no data rows
            if row_count == 0:
                self.warnings.append("CSV file has headers but no data rows")
            
            # Return validation result
            return len(self.errors) == 0, self.errors, self.warnings
            
        except UnicodeDecodeError:
            self.errors.append("CSV file has encoding issues. Please ensure it's UTF-8 encoded.")
            return False, self.errors, self.warnings
        except Exception as e:
            self.errors.append(f"Error validating CSV file: {str(e)}")
            return False, self.errors, self.warnings
    
    def _validate_row(self, row: Dict[str, str], row_number: int) -> None:
        """
        Validate a single row of data.
        
        Args:
            row: Dictionary containing row data
            row_number: Row number for error reporting
        """
        # Check for required field values
        for field in self.REQUIRED_FIELDS:
            if not row.get(field):
                self.errors.append(f"Row {row_number}: Missing required value for '{field}'")
        
        # Validate numeric fields
        for field in self.NUMERIC_FIELDS:
            if field in row and row[field]:
                try:
                    float(row[field])
                except ValueError:
                    self.errors.append(f"Row {row_number}: Field '{field}' should be numeric, got '{row[field]}'")
        
        # Validate date fields
        for field in self.DATE_FIELDS:
            if field in row and row[field]:
                valid = self._is_valid_date(row[field])
                if not valid:
                    self.warnings.append(f"Row {row_number}: Field '{field}' has an unrecognized date format: '{row[field]}'")
        
        # Validate boolean fields
        for field in self.BOOLEAN_FIELDS:
            if field in row and row[field]:
                valid = self._is_valid_boolean(row[field])
                if not valid:
                    self.errors.append(
                        f"Row {row_number}: Field '{field}' should be a boolean value, got '{row[field]}'"
                    )
    
    def _is_valid_date(self, date_str: str) -> bool:
        """
        Check if a string is a valid date in various formats.
        
        Args:
            date_str: Date string to validate
            
        Returns:
            Boolean indicating if date is valid
        """
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M',
            '%m/%d/%Y',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M',
            '%d/%m/%Y',
        ]
        
        for fmt in formats:
            try:
                datetime.datetime.strptime(date_str, fmt)
                return True
            except ValueError:
                continue
                
        return False
    
    def _is_valid_boolean(self, bool_str: str) -> bool:
        """
        Check if a string represents a valid boolean value.
        
        Args:
            bool_str: String to validate as boolean
            
        Returns:
            Boolean indicating if string is a valid boolean representation
        """
        if not isinstance(bool_str, str):
            return isinstance(bool_str, bool)
            
        valid_values = ['0', '1', 'true', 'false', 'yes', 'no', 't', 'f', 'y', 'n']
        return bool_str.lower() in valid_values