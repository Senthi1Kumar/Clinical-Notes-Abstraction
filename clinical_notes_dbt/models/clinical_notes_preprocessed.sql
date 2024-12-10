{{ config(
    materialized='table',
    schema='public'
) }}

-- Dynamically identify text columns (excluding idx)
WITH column_info AS (
    SELECT 
        column_name, 
        data_type
    FROM information_schema.columns
    WHERE 
        table_name = 'clinical_notes' 
        AND table_schema = 'public'
        AND column_name != 'idx'
        AND data_type IN ('text', 'character varying', 'character')
),

-- Generate dynamic lowercase transformation
transformed_notes AS (
    SELECT 
        idx,  -- Preserve original patient ID
        
        -- Dynamically lowercase all text columns
        {% for column in column_info %}
        LOWER({{ column.column_name }}) AS {{ column.column_name }}_lowercase{% if not loop.last %},{% endif %}
        {% endfor %}
    FROM 
        {{ source('clinical_notes_source', 'clinical_notes') }}
)

SELECT 
    idx,
    {% for column in column_info %}
    {{ column.column_name }}_lowercase{% if not loop.last %},{% endif %}
    {% endfor %}
FROM 
    transformed_notes