import xml.etree.ElementTree as ET
import pandas as pd

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np


def vtu_to_dataframe(vtu_content):
    """
    Convert a VTU file content to pandas DataFrames for both field data and point data.

    Args:
        vtu_content (str): The content of the VTU file as a string

    Returns:
        tuple: (field_data_df, point_data_df) where:
            - field_data_df: DataFrame with field data (metadata)
            - point_data_df: DataFrame with point data (particle information)
    """
    # Parse the XML content
    root = ET.fromstring(vtu_content)

    # Process FieldData (metadata)
    field_data = root.find('.//FieldData')
    field_data_dict = {}
    if field_data is not None:
        for array in field_data.findall('DataArray'):
            name = array.attrib['Name']
            value = array.text.strip() if array.text else ''
            try:
                dtype = array.attrib.get('type', 'Float64')
                if dtype.startswith('Float'):
                    field_data_dict[name] = float(value)
                elif dtype.startswith('Int') or dtype.startswith('UInt'):
                    field_data_dict[name] = int(value)
                else:
                    field_data_dict[name] = value
            except (ValueError, AttributeError):
                field_data_dict[name] = value

    field_data_df = pd.DataFrame(
        [field_data_dict]) if field_data_dict else pd.DataFrame()

    # Process PointData (particle information)
    point_data = root.find('.//PointData')
    point_data_dict = {}
    if point_data is not None:
        for array in point_data.findall('DataArray'):
            name = array.attrib['Name']
            num_components = int(array.attrib.get('NumberOfComponents', '1'))

            # Parse the data values
            text = array.text.strip() if array.text else ''
            # Split and remove empty strings
            values = [v for v in text.split() if v]

            # Convert to appropriate numeric type
            dtype = array.attrib.get('type', 'Float64')
            if dtype.startswith('Float'):
                values = [float(v) for v in values]
            elif dtype.startswith('Int') or dtype.startswith('UInt'):
                values = [int(v) for v in values]

            # Reshape based on number of components
            if num_components > 1:
                values = np.array(values).reshape(-1, num_components)
                for i in range(num_components):
                    letter = ["x", "y", "z", "w"][i]
                    point_data_dict[f"{name}_{letter}"] = values[:, i]
            else:
                point_data_dict[name] = values

    # Process Points (coordinates)
    points = root.find('.//Points/DataArray')
    if points is not None and points.text:
        text = points.text.strip()
        values = [float(v) for v in text.split() if v]
        values = np.array(values).reshape(-1, 3)
        point_data_dict['x'] = values[:, 0]
        point_data_dict['y'] = values[:, 1]
        point_data_dict['z'] = values[:, 2]

    point_data_df = pd.DataFrame(
        point_data_dict) if point_data_dict else pd.DataFrame()

    # combine field_data_df and point_data_df
    df = pd.concat([field_data_df, point_data_df], axis=1)

    return df
