dynamic_few_shots_wikidata = \
  {
    'name': 'DFS-WDT',
    'separator': '###',
    'open_tag': '<SPARQL>',
    'close_tag': '</SPARQL>',
    'instruction':"""
The task involves translating questions from English into SPARQL queries for the Wikidata knowledge graph. The queries must follow specific guidelines to ensure accuracy and correct execution:
1. Enclose SPARQL queries within <SPARQL></SPARQL> tags.
2. Utilize all provided golden entities and relations exclusively to construct the query accurately. Do not use any other entities or relations. 
3. Examples are provided below for guidance.

Examples:
"""
  }