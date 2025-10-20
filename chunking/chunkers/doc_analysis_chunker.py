import base64
import fitz
import json
import logging
import os
import re
import requests
import markdown
from io import BytesIO
from markdownify import markdownify as md
from bs4 import BeautifulSoup
from PIL import Image
import time
from urllib.parse import urlparse, unquote
from utils.file_utils import get_secret

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

from .base_chunker import BaseChunker
from ..exceptions import UnsupportedFormatError
from tools import DocumentIntelligenceClient
from tools.mistral import MistralPixtralClient
from tools.content_understanding import ContentUnderstandingService


class DocAnalysisChunker(BaseChunker):
    """
    DocAnalysisChunker class is responsible for analyzing and splitting document content into chunks
    based on specific format criteria, utilizing the Document Intelligence service for content analysis.

    Format Support:
    ---------------
    The DocAnalysisChunker class leverages the Document Intelligence service to process and analyze
    a wide range of document formats. The class ensures that document content is accurately processed
    and divided into manageable chunks.

    - Supported Formats: The chunker processes document formats supported by the Document Intelligence client.
    - Unsupported Formats: If a document's format is not supported by the client, an `UnsupportedFormatError` is raised.

    Chunking Parameters:
    --------------------
    - max_chunk_size: The maximum size of each chunk in tokens. This value is sourced from the `NUM_TOKENS` 
    environment variable, with a default of 2048 tokens.
    - token_overlap: The number of overlapping tokens between consecutive chunks, sourced from the `TOKEN_OVERLAP` 
    environment variable, with a default of 100 tokens.
    - minimum_chunk_size: The minimum size of each chunk in tokens, sourced from the `MIN_CHUNK_SIZE` environment 
    variable, with a default of 100 tokens.

    Document Analysis:
    ------------------
    - The document is analyzed using the Document Intelligence service, extracting its content and structure.
    - The analysis process includes identifying the number of pages and providing a preview of the content.
    - If the document is large, a warning is logged to indicate potential timeout issues during processing.

    Content Chunking:
    -----------------
    - The document content is split into chunks using format-specific strategies.
    - HTML tables in the content are replaced with placeholders during the chunking process to simplify splitting.
    - After chunking, the original content, such as HTML tables, is restored in place of the placeholders.
    - The chunking process also manages page numbering based on the presence of page breaks, ensuring each chunk 
    is correctly associated with its corresponding page.

    Error Handling:
    ---------------
    - The class includes comprehensive error handling during document analysis, such as managing unsupported formats 
    and handling general exceptions.
    - The chunking process's progress and outcomes, including the number of chunks created or skipped, are logged.
    """


    def __init__(self, data, max_chunk_size=None, minimum_chunk_size=None, token_overlap=None):
        super().__init__(data)
        self.max_chunk_size = max_chunk_size or int(os.getenv("NUM_TOKENS", "2048"))
        self.minimum_chunk_size = minimum_chunk_size or int(os.getenv("MIN_CHUNK_SIZE", "100"))
        self.token_overlap = token_overlap or int(os.getenv("TOKEN_OVERLAP", "100"))
        self.docint_client = DocumentIntelligenceClient(document_filename=self.filename)
        self.supported_formats = self.docint_client.file_extensions
        # Feature flags
        self.enable_pixtral = os.getenv("ENABLE_PIXTRAL_OCR", "false").lower() == "true"
        self.enable_cu = os.getenv("ENABLE_CONTENT_UNDERSTANDING", "false").lower() == "true"
        self.pixtral_client = MistralPixtralClient() if self.enable_pixtral else None

    def get_tables(self):
        if self.extension not in self.supported_formats:
            raise UnsupportedFormatError(f"[doc_analysis_chunker] {self.extension} format is not supported")

        logging.info(f"[doc_analysis_chunker][{self.filename}] Running get_tables.")
        tables = []
        document, analysis_errors = self._analyze_document_with_retry()
        if analysis_errors:
            formatted_errors = ', '.join(map(str, analysis_errors))
            raise Exception(f"Error in doc_analysis_chunker analyzing {self.filename}: {formatted_errors}")

        doc_analysis = {
            "content": document["content"]
        }
        if "tables" in document:
            doc_analysis["tables"] = document["tables"] 
            tables = self._process_document_tables(self.url, document["tables"])
        return doc_analysis["content"], tables

    def understand_tables(self):
        """Analyze and understand tables: OCR + Content Understanding + artifact publishing."""
        if self.extension not in self.supported_formats:
            raise UnsupportedFormatError(f"[doc_analysis_chunker] {self.extension} format is not supported")

        logging.info(f"[doc_analysis_chunker][{self.filename}] Running understand_tables.")
        if not (self.enable_pixtral and self.enable_cu):
            raise RuntimeError("Table understanding requires ENABLE_PIXTRAL_OCR and ENABLE_CONTENT_UNDERSTANDING=true")

        document, analysis_errors = self._analyze_document_with_retry()
        if analysis_errors:
            formatted_errors = ', '.join(map(str, analysis_errors))
            raise Exception(f"Error in doc_analysis_chunker analyzing {self.filename}: {formatted_errors}")

        content = document.get("content", "")
        tables = []
        if "tables" in document:
            tables = self._process_document_tables(self.url, document["tables"])  # [{'name','url'}]

        # Build mapping-like structure for understanding
        mapping = []
        for i, t in enumerate(tables):
            mapping.append({
                "name": t["name"],
                "url": t["url"],
                "html_table_content": "",  # Unknown in this flow
                "table_index": i
            })

        enriched = self._run_table_understanding(mapping) if mapping else []

        return content, enriched
    
    def get_chunks(self, document):

        ## Obtain the mapping between table image name, table image URL and HTML table content
        table_html_table_url_map = self._map_html_tables_with_url(document)

        ## Optionally run Table Understanding (Pixtral OCR + Content Understanding)
        if self.enable_pixtral and self.enable_cu and table_html_table_url_map:
            try:
                table_html_table_url_map = self._run_table_understanding(table_html_table_url_map)
            except Exception as e:
                logging.error(f"[doc_analysis_chunker][{self.filename}] Table understanding step failed: {str(e)}")

        ## Replace the HTML tables with the URL of the table image
        document['content'] = self._replace_html_tables_with_url(document, table_html_table_url_map)

        ## Upload to storage account container a .txt with the HTML table content
        self._save_html_tables_in_storage(table_html_table_url_map)

        ## Perform chunking
        chunks = self._process_document_chunks(document, table_html_table_url_map)
        
        return chunks

    def _is_table_complex(self, html_table: str) -> bool:
        """Heurística básica para identificar tablas complejas."""
        try:
            soup = BeautifulSoup(html_table, 'html.parser')
            th_count = len(soup.find_all('th'))
            has_rowspan = bool(soup.find(attrs={"rowspan": True}))
            has_colspan = bool(soup.find(attrs={"colspan": True}))
            # Borderless heuristic
            border_attr = soup.find('table').get('border') if soup.find('table') else None
            style_attr = soup.find('table').get('style') if soup.find('table') else ''
            borderless = (border_attr in (None, '0', 0)) and ('border' not in style_attr or 'none' in style_attr)
            multi_header_rows = False
            thead = soup.find('thead')
            if thead and len(thead.find_all('tr')) > 1:
                multi_header_rows = True
            return has_rowspan or has_colspan or borderless or multi_header_rows or th_count > 10
        except Exception:
            return False

    def _run_table_understanding(self, mapping):
        """
        Ejecuta Pixtral OCR + Content Understanding sobre cada tabla y publica artefactos.
        Enriquecerá cada item del mapeo con:
          - normalized_md (markdown)
          - artifacts: {json_url, csv_url, md_url, schema_url, semantic_url}
        """
        cu = ContentUnderstandingService(document_filename=self.filename)
        enriched = []
        for i, item in enumerate(mapping):
            url = item.get('url')
            name = item.get('name')
            html = item.get('html_table_content', '')

            # Ejecutar sólo si la tabla es compleja (o si se desea siempre)
            if not self._is_table_complex(html):
                enriched.append(item)
                continue

            logging.info(f"[doc_analysis_chunker][{self.filename}] Understanding complex table {i+1}: {name}")
            try:
                # Descargar imagen de tabla
                # Generar SAS si es necesario y descargar bytes
                table_url_with_sas = self.blob_client.generate_sas_token(url)
                table_img = self.blob_client.download_blob(table_url_with_sas)

                # Pixtral OCR → estructura
                ocr = self.pixtral_client.analyze_table_image(table_img) if self.pixtral_client else {}

                # Content Understanding → normalización y artefactos in-memory
                cu_result = cu.process(ocr)

                # Subir artefactos a blob
                base_name = os.path.splitext(os.path.basename(url))[0]
                artifacts = {}

                # Derive a stable table_id from base name
                item['table_id'] = base_name

                # JSON OCR
                json_bytes = json.dumps(ocr, ensure_ascii=False).encode('utf-8')
                artifacts['json_url'] = self._upload_bytes(f"{base_name}.ocr.json", json_bytes)

                # CSV
                csv_buf = cu_result['csv_bytes']  # BytesIO
                csv_buf.seek(0)
                artifacts['csv_url'] = self._upload_bytes(f"{base_name}.csv", csv_buf.read())

                # Markdown
                md_text = cu_result['markdown']
                artifacts['md_url'] = self._upload_bytes(f"{base_name}.md", md_text.encode('utf-8'))

                # Schema
                schema_bytes = json.dumps(cu_result['schema'], ensure_ascii=False).encode('utf-8')
                artifacts['schema_url'] = self._upload_bytes(f"{base_name}.schema.json", schema_bytes)

                # Semantic
                semantic_bytes = json.dumps(cu_result['semantic'], ensure_ascii=False).encode('utf-8')
                artifacts['semantic_url'] = self._upload_bytes(f"{base_name}.semantic.json", semantic_bytes)

                # Enriquecer mapping item
                item['artifacts'] = artifacts
                item['normalized_md'] = md_text
                item['quality_confidence'] = cu_result.get('quality_confidence', 0.0)
                # Expose header hierarchy for downstream chunk metadata
                try:
                    item['header_hierarchy'] = cu_result['semantic'].get('header_hierarchy', [])
                except Exception:
                    item['header_hierarchy'] = []

            except Exception as e:
                logging.error(f"[doc_analysis_chunker][{self.filename}] Understanding failed for table {name}: {str(e)}")
            enriched.append(item)

        return enriched

    def _upload_bytes(self, blob_name: str, data: bytes) -> str:
        from io import BytesIO as _BytesIO

        buf = _BytesIO(data)
        buf.seek(0)
        return self.blob_client.upload_blob(blob_name, buf)

    def _map_html_tables_with_url(self, document):
        '''
        1. Detects HTML tables in document content
        2. Matches them by ordinal position to documentTables (list of dictionaries created in table-extraction)
        3. Creates a new list of dicts with the mapping between documentTables and it's related HTML table from content
        '''

        # Get the content of the document
        document_content = document['documentContent']
        # Check if the document has documentTables set (tables associated with the document)
        # If not, set to empty list
        if 'documentTables' in document:
            document_tables = document['documentTables']
        else:
            document_tables = []

            return []

        logging.info(f"[doc_analysis_chunker][{self.filename}] Starting HTML table detection")
        logging.info(f"[doc_analysis_chunker][{self.filename}] Document tables provided: {len(document_tables)}")

        ### 1. Match all the HTML tables in the content 
        try:
            table_pattern = r"<table>.*?</table>"
            html_tables = re.findall(table_pattern, document_content, re.DOTALL)
            
            logging.info(f"[doc_analysis_chunker][{self.filename}] Found {len(html_tables)} HTML tables in document content")
        except Exception as e:
            logging.error(f"[doc_analysis_chunker][{self.filename}] Error parsing HTML tables from content: {str(e)}")
            raise

        ### 2. Create mapping

        html_document_table_mapping = []

        # Validation: Check if counts match between provided documentTables and detected tables

        if len(html_tables) != len(document_tables):
            logging.error(f"[doc_analysis_chunker][{self.filename}] MISMATCH: Found {len(html_tables)} HTML tables but {len(document_tables)} document table entries")
            raise Exception(f"Mismatch between provided tables array and the ones detected in the content")
        else:
            logging.info(f"[doc_analysis_chunker][{self.filename}] - HTML tables count matches document tables count: {len(html_tables)}")

        # Iterate over the detected tables 
        for i in range(len(html_tables)):
            html_table_content = html_tables[i]
            # Get table N from document_tables
            table_info = document_tables[i]
            
            # Create enhanced table entry
            mapping_item = {
                "name": table_info["name"],
                "url": table_info["url"],
                "html_table_content": html_table_content,
                "table_index": i
            }
            
            html_document_table_mapping.append(mapping_item)

        return html_document_table_mapping

    def _replace_html_tables_with_url(self, document, enhanced_document_tables):
        '''
        Replaces HTML tables with URL markers in content
        '''

        # Get the content of the document
        modified_content = document['documentContent']
        # Check if the mapping has been set (this mean, this document has tables)
        if not enhanced_document_tables:
            logging.info(f"[doc_analysis_chunker][{self.filename}] No tables to be replaced. Mapping array is empty")
            ## If mapping not exists, then, do not replace content
            return modified_content
    
        logging.info(f"[doc_analysis_chunker][{self.filename}] Starting HTML table replacement with table image URL")
        
        try:
            # Process tables in reverse order to maintain positions
            for i in reversed(range(len(enhanced_document_tables))):
                table_info = enhanced_document_tables[i]
                html_table_content = table_info["html_table_content"]
                table_url = table_info["url"]
                
                # Create URL marker
                url_marker = f"[TABLE_IMAGE_URL:{table_url}]"
                
                # Replace the HTML table with the URL marker
                # Use the exact HTML content as stored in enhanced_document_tables
                if html_table_content in modified_content:
                    modified_content = modified_content.replace(html_table_content, url_marker, 1)  # Replace only first occurrence
                    logging.info(f"[doc_analysis_chunker][{self.filename}] Replaced table {i+1} with URL marker: {table_info['name']}")
                else:
                    logging.warning(f"[doc_analysis_chunker][{self.filename}] Could not find HTML table {i+1} in content for replacement")
        except Exception as e:
            logging.error(f"[doc_analysis_chunker][{self.filename}] Error replacing HTML tables with URL markers: {str(e)}")
            raise Exception(f"Error replacing HTML tables with URL markers: {str(e)}")
        
        return modified_content
    
    def _save_html_tables_in_storage(self, mapping_table_image_url):
        '''
        Create a .txt file to save the HTML tables. Then, upload it to the storage account.
        '''

        # Save the HTML tables to a .txt file for debugging purposes
        try:
            parsed_url = urlparse(self.file_url)
            container_name = parsed_url.path.split("/")[1]
            url_decoded = unquote(parsed_url.path)
            blob_name = url_decoded[len(container_name) + 2:]
            blob_name = f"{os.path.splitext(blob_name)[0]}.txt"
            output_txt_file = f"/tmp/{blob_name}"

            with open(output_txt_file, "w", encoding="utf-8") as file:
                for i, table_info in enumerate(mapping_table_image_url):
                    file.write(F"TABLA {i + 1}:\n")
                    file.write(table_info['html_table_content'] + "\n\n\n")

            with open(output_txt_file, "rb") as data:
                blob_url = self.blob_client.upload_blob(blob_name, data)
            
            logging.info(f"Table content stored in {blob_url}")

            if os.path.exists(output_txt_file):
                os.remove(output_txt_file)
        except Exception as e:
            logging.error(f"[doc_analysis_chunker][{self.filename}] Error saving HTML tables to blob storage: {str(e)}")

            if os.path.exists(output_txt_file):
                os.remove(output_txt_file)

            raise

    def _save_html_tables(self, content):

        # Match all the HTML tables in the content 
        try:
            table_pattern = r"<table>.*?</table>"
            html_tables = re.findall(table_pattern, content, re.DOTALL)
            
            logging.info(f"[doc_analysis_chunker][{self.filename}] Found {len(html_tables)} HTML tables in document content")
        except Exception as e:
            logging.error(f"[doc_analysis_chunker][{self.filename}] Error parsing HTML tables from content: {str(e)}")
            raise

        # Save the HTML tables to a .txt file for debugging purposes
        try:
            index = 0
            parsed_url = urlparse(self.file_url)
            container_name = parsed_url.path.split("/")[1]
            url_decoded = unquote(parsed_url.path)
            blob_name = url_decoded[len(container_name) + 2:]
            blob_name = f"{os.path.splitext(blob_name)[0]}.txt"
            output_txt_file = f"/tmp/{blob_name}"

            with open(output_txt_file, "w", encoding="utf-8") as file:
                for index, table_html in enumerate(html_tables):
                    file.write(F"TABLA {index + 1}:\n")
                    file.write(table_html + "\n\n\n")

            with open(output_txt_file, "rb") as data:
                blob_url = self.blob_client.upload_blob(blob_name, data)
            
            logging.info(f"Table content stored in {blob_url}")

            if os.path.exists(output_txt_file):
                os.remove(output_txt_file)
        except Exception as e:
            logging.error(f"[doc_analysis_chunker][{self.filename}] Error saving HTML tables to blob storage: {str(e)}")

            if os.path.exists(output_txt_file):
                os.remove(output_txt_file)

            raise

    def _replace_table_contents(self, content, tables, table_descriptions):
        def replacer(match):
            nonlocal index
            if index < len(tables):
                # json_str = json.dumps(tables[index]["description"], indent=2)  # Convert JSON to string
                # replacement_html = f"<table>{json_str}</table>"
                table_url = tables[index]["url"]
                table_desc = table_descriptions[index]
                replacement_html = f"{table_desc}"
                index += 1
                return replacement_html
            return match.group(0)

        index = 0
        updated_html = re.sub(r"<table>.*?</table>", replacer, content, flags=re.DOTALL)

        parsed_url = urlparse(self.file_url)
        container_name = parsed_url.path.split("/")[1]
        url_decoded = unquote(parsed_url.path)
        blob_name = url_decoded[len(container_name) + 2:]
        blob_name = f"{os.path.splitext(blob_name)[0]}.txt"
        output_txt_file = f"/tmp/{blob_name}"

        with open(output_txt_file, "w", encoding="utf-8") as file:
            for index, table in enumerate(table_descriptions):
                file.write(F"TABLA {index + 1}:\n")
                file.write(table + "\n\n\n")

        with open(output_txt_file, "rb") as data:
            blob_url = self.blob_client.upload_blob(blob_name, data)
        
        logging.info(f"Table analysis stored in {blob_url}")

        if os.path.exists(output_txt_file):
            os.remove(output_txt_file)

        return updated_html


    def _analyze_document_with_retry(self, retries=3):
        """
        Analyzes the document using the Document Intelligence Client, with a retry mechanism for error handling.

        Args:
            retries (int): The number of times to retry the document analysis in case of failure. Defaults to 3 retries.

        Returns:
            tuple: A tuple containing the analyzed document content and any analysis errors encountered.

        Raises:
            Exception: If the document analysis fails after the specified number of retries.
        """
        for attempt in range(retries):
            try:
                document, analysis_errors = self.docint_client.analyze_document(self.file_url)
                return document, analysis_errors
            except Exception as e:
                logging.error(f"[doc_analysis_chunker][{self.filename}] docint analyze document failed on attempt {attempt + 1}/{retries}: {str(e)}")
                if attempt == retries - 1:
                    raise
        return None, None
    

    def _process_document_tables(self, document, tables=[]):
        if len(tables) == 0:
            pass
        local_file_path, blob_name = self.blob_client.download_blob_locally(document)
        image_list = self._extract_and_replace_images(local_file_path, blob_name, tables)
        #table_desc = self._generate_table_desc(image_list[1])
        #image_list[1]["description"] = table_desc

        return image_list


    def _extract_and_replace_images(self, local_pdf, filename, tables):
        image_list = []
        blob_name = os.path.splitext(filename)[0]

        doc = fitz.open(local_pdf)
        dpi = 300
    
        for table_idx, table in enumerate(tables):
            page_index = table["boundingRegions"][0]["pageNumber"] - 1  # Convert to 0-based index
            page = doc.load_page(page_index)

            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            # Render page to a PNG image
            pix = page.get_pixmap(matrix=mat)
            # Save the image to a temporary file
            output_path = f"/tmp/{blob_name}_page_{page_index}.png"
            pix.save(output_path)
            
            # Open the image using Pillow
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Get bounding box (Azure gives in inches)
            bbox = table["boundingRegions"][0]["polygon"] 

            # Create a list of tuples for the rectangle points
            polygon_pixels = [(x * dpi, y * dpi) for x, y in zip(bbox[0::2], bbox[1::2])]
            # Draw the rectangle (bounding box)

            x_coords, y_coords = zip(*polygon_pixels)
            left, top = int(min(x_coords)), int(min(y_coords))
            right, bottom = int(max(x_coords)), int(max(y_coords))

            # Crop the image to the bounding box
            cropped_img = image.crop((left, top, right, bottom))

            # Save the cropped image
            buffered = BytesIO()
            cropped_img.save(buffered, format="PNG")
            buffered.seek(0)
            #img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8") 

            #Subir imagen
            image_name = f"{blob_name}_table_{str(table_idx).zfill(2)}.png"
            image_url = self.blob_client.upload_blob(image_name, buffered)

            #Borrar imagen

            #Guardar lista
            image_list.append({
                "name": image_name,
                "url": image_url
            })  

            # Remove generated page png
            if os.path.exists(output_path):
                os.remove(output_path)
        
        doc.close()

        # Remove PDF file
        if os.path.exists(local_pdf):
            os.remove(local_pdf)

        return image_list


    def generate_table_desc(self, data):

        table_url = self.blob_client.generate_sas_token(data["tableUrl"])

        blob_data = self.blob_client.download_blob(table_url)

        # Prefer Foundry vision if enabled
        try:
            use_foundry = os.getenv('ENABLE_FOUNDRY', 'false').lower() == 'true'
        except Exception:
            use_foundry = False
        if use_foundry:
            try:
                from tools.foundry import FoundryVisionClient
                from tools.content_understanding import ContentUnderstandingService
                vision = FoundryVisionClient()
                ocr = vision.analyze_table_image(blob_data)
                cu = ContentUnderstandingService(document_filename=self.filename)
                result = cu.process(ocr)
                md = result.get('markdown', '')
                return md
            except Exception as e:
                logging.error(f"[doc_analysis_chunker]{self.filename}-{data['tableUrl']} foundry table_desc error: {str(e)}")

        img_base64 = base64.b64encode(blob_data).decode('utf-8')

        prompt = f'''
        Actúa como un experto en reformateo de tablas.
 
        1- Voy a proporcionarte una tabla, algunas celdas pueden estar combinadas tanto horizontal como verticalmente.
        
        2- Tu tarea consiste en convertir la tabla en una versión simple, sin ninguna celda combinada. Para ello:
        
        - Identifica claramente cada columna y cada fila.
        - Cuando encuentres una celda combinada verticalmente (que abarque varias filas), repite el contenido de esa celda en cada fila que cubra esa combinación.
        - Cuando encuentres una celda combinada horizontalmente (que abarque varias columnas), repite el contenido de esa celda en cada columna correspondiente.
        - Utiliza las técnicas de: detección de bordes y líneas, detección de regiones contiguas, detección de bordes con Canny, detección de bordes con Umbral Adaptativo.
        
        3- Asegúrate de mantener el orden lógico de la tabla y la relación que cada valor tenía con su fila y columna original.
        
        4- No pierdas los encabezados (si existen) y procura que cada columna tenga un título si es que la tabla original lo trae.
        
        5- Si los encabezados están en varias filas o tienen celdas combinadas, repetir el valor de la celda superior en la celda inferior separando los valores por punto (.).
        
        6- Devuelve la tabla final en formato Markdown. No incluyas celdas combinadas en tu respuesta final.
        
        7- Si hay notas, leyendas o referencias a secciones, por favor mantén esos textos en su respectiva fila/columna (repitiéndolos en todas las filas/columnas si también estaban combinadas).
        
        8- Si detectas que una celda está combinada verticalmente (por ejemplo, va desde la fila i hasta la fila j), deberás copiar el contenido de esa celda en cada una de las filas entre i y j en la columna correspondiente.
        
        9- Si detectas que una celda está combinada horizontalmente (por ejemplo, va desde la columna k hasta la columna l), deberás copiar el contenido de esa celda en cada una de las columnas entre k y l en la misma fila.
        
        10- Si la fusión alcanza la última fila o columna, no dejes esa posición vacía: repite el contenido ahí también.
        
        11- Utiliza la información de coordenadas y dimensiones de cada celda (o su bounding box) para determinar con precisión qué filas o columnas ocupa. Cualquier fila/columna parcialmente cubierta por esa celda fusionada debe recibir el mismo texto.
        
        12- Si dentro de esa celda fusionada hay textos con notas, leyendas o referencias, esos mismos deben replicarse en todas las celdas ‘resultantes’ al descombinar.
        
        13- IMPORTANTE. Revisa cuidadosamente que los valores de las celdas combinadas se repitan en cada celda simple resultante.
        
        Mostrar la tabla unicamente, sin ningun tipo de explicación
        '''

        input_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": 
             [
                { "type": "text", "text": f"{prompt}" },
                { "type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]}
        ]
        openai_service_name = os.environ.get("AZURE_OPENAI_SERVICE_NAME")

        endpoint = f"https://{openai_service_name}.openai.azure.com/"
        deployment_name = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT")  # Replace with your model deployment name

        # Define the API version
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION")

        # Set the URL for the Chat Completions API
        url = f"{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"

        # Define the request payload
        payload = {
            "messages": input_messages
        }

        api_key = get_secret("azureOpenAIKey")

        # Define headers
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }
        start_time = time.time()
        # Make the request
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response_data = response.json()
            response_time = time.time() - start_time
            logging.info(f"Finished table summarization. {round(response_time,2)} seconds")
        except Exception as e:
            logging.error(f"[doc_analysis_chunker]{self.filename}-{data['tableUrl']} get_completion Error: {str(e)}")

        if response_data["choices"][0]["finish_reason"] == "content_filter":
            completion = ""
            return completion

        completion = response_data["choices"][0]["message"]["content"]

        return completion


    def _process_document_chunks(self, document, html_table_url_table_mapping):
        """
        Processes the analyzed document content into manageable chunks.

        Args:
            document (dict): The analyzed document content provided by the Document Intelligence Client.

        Returns:
            list: A list of dictionaries, where each dictionary represents a processed chunk of the document content.

        The method performs the following steps:
        1. Prepares the document content for chunking, including numbering page breaks.
        2. Splits the content into chunks using a chosen splitting strategy.
        3. Iterates through the chunks, determining their page numbers and creating chunk representations.
        4. Skips chunks that do not meet the minimum size requirement.
        5. Logs the number of chunks created and skipped.
        """
        chunks = []

        # Get the content of the document
        document_content = document['content']

        # Check if a mapping between HTML tables and URL tables has been set. This means, document has tables
        # If not, set to empty list
        url_to_content_mapping = {}
        url_to_artifacts_mapping = {}
        url_to_meta_mapping = {}

        if html_table_url_table_mapping:
            # Prefer normalized markdown when available; fallback to original HTML
            for table_info in html_table_url_table_mapping:
                url = table_info['url']
                normalized_md = table_info.get('normalized_md')
                html_backup = table_info.get('html_table_content', '')
                url_to_content_mapping[url] = normalized_md if normalized_md else html_backup
                if 'artifacts' in table_info:
                    url_to_artifacts_mapping[url] = table_info['artifacts']
                # meta: table_id and header_hierarchy
                url_to_meta_mapping[url] = {
                    'table_id': table_info.get('table_id'),
                    'header_hierarchy': table_info.get('header_hierarchy', [])
                }
        
        document_content = self._number_pagebreaks(document_content)

        text_chunks = self._chunk_content(document_content)
        chunk_id = 0
        skipped_chunks = 0
        current_page = 1

        for text_chunk, chunk_headers, num_tokens in text_chunks:
            current_page = self._update_page(text_chunk, current_page)
            chunk_page = self._determine_chunk_page(text_chunk, current_page)
            chunk_id += 1

            # Extract table image URLs for this chunk
            chunk_table_image_urls = self._extract_table_urls_for_chunk(text_chunk)

            # Replace URL markers with the corresponding content (normalized MD or HTML)
            restored_text_chunk = self._restore_tables_from_markers(text_chunk, url_to_content_mapping)

            # Collect related files (artifacts) referenced in this chunk
            related_files = self._collect_related_files_for_chunk(text_chunk, url_to_artifacts_mapping)
            # Collect table semantics and ids referenced in this chunk
            table_ids, header_hiers = self._collect_table_semantics_for_chunk(text_chunk, url_to_meta_mapping)

            chunk = self._create_chunk(
                chunk_id=chunk_id,
                content=restored_text_chunk,
                page=chunk_page,
                headers=chunk_headers,
                related_images=chunk_table_image_urls,
                related_files=related_files,
                table_ids=table_ids
            )
            if header_hiers:
                chunk['tableHeaderHierarchies'] = header_hiers
            chunks.append(chunk)

        logging.info(f"[doc_analysis_chunker][{self.filename}] {len(chunks)} chunk(s) created")
        if skipped_chunks > 0:
            logging.info(f"[doc_analysis_chunker][{self.filename}] {skipped_chunks} chunk(s) skipped")
        return chunks

    def _extract_table_urls_for_chunk(self, chunk_content):
        """
        Extract table image URLs that correspond to HTML tables in the current chunk
        """
        table_urls = []
        
        try:
            # Find all table markers in the chunk
            marker_pattern = r'\[TABLE_IMAGE_URL:([^\]]+)\]'
            # Extract just the matched URLs
            urls = re.findall(marker_pattern, chunk_content)
            
            table_urls = urls
            
        except Exception as e:
            raise Exception(f"Error replacing HTML tables with URL markers: {str(e)}")
        
        return table_urls
    
    def _restore_tables_from_markers(self, chunk_content, url_to_content_mapping):
        """
        Restore HTML tables from table URL image markers in this chunk content.
        """

        restored_content = chunk_content
        
        try:
            # Find all URL markers in the content
            marker_pattern = r'\[TABLE_IMAGE_URL:([^\]]+)\]'
            markers_found = re.findall(marker_pattern, chunk_content)
            
            #################################################################################

            # Case 1: No markers found in this chunk
            if len(markers_found) == 0:
                return chunk_content
                
            elif len(markers_found) == 1:
                # Case 2: Single marker
                url = markers_found[0]
                if url in url_to_content_mapping:
                    marker_to_replace = f"[TABLE_IMAGE_URL:{url}]"
                    content = url_to_content_mapping[url]
                    restored_content = chunk_content.replace(marker_to_replace, content)
                else:
                    # Return original marker if no mapping found
                    raise Exception(f"A single marker could not be replaced.")
                    
            else:
                # Case 3: Multiple markers - replace all with their corresponding HTML tables
                def replace_marker(match):
                    url = match.group(1)
                    if url in url_to_content_mapping:
                        return url_to_content_mapping[url]
                    else:
                        # Return original marker if no mapping found
                        return match.group(0)
                
                # Replace all markers with their corresponding HTML content
                restored_content = re.sub(marker_pattern, replace_marker, chunk_content)
                
                # Verify all markers were processed
                remaining_markers = re.findall(marker_pattern, restored_content)
                
                if len(remaining_markers) != 0:
                    raise Exception(f"There are {len(remaining_markers)} markers left to be replaced.")
            
        except Exception as e:
            raise Exception(f"Error while replacing markers in chunk. Error: {str(e)}")
        
        return restored_content

    def _collect_related_files_for_chunk(self, chunk_content, url_to_artifacts_mapping):
        """Collect artifact URLs for tables referenced by markers in this chunk."""
        related_files = []
        try:
            marker_pattern = r'\[TABLE_IMAGE_URL:([^\]]+)\]'
            markers_found = re.findall(marker_pattern, chunk_content)
            for url in markers_found:
                artifacts = url_to_artifacts_mapping.get(url)
                if artifacts:
                    # Add all artifact URLs
                    for key in [
                        'json_url', 'csv_url', 'md_url', 'schema_url', 'semantic_url'
                    ]:
                        if artifacts.get(key):
                            related_files.append(artifacts[key])
        except Exception:
            pass
        return related_files

    def _collect_table_semantics_for_chunk(self, chunk_content, url_to_meta_mapping):
        """Return (table_ids, header_hierarchies) referenced in this chunk."""
        table_ids = []
        header_hiers = []
        try:
            marker_pattern = r'\[TABLE_IMAGE_URL:([^\]]+)\]'
            markers_found = re.findall(marker_pattern, chunk_content)
            for url in markers_found:
                meta = url_to_meta_mapping.get(url)
                if not meta:
                    continue
                tid = meta.get('table_id')
                if tid:
                    table_ids.append(tid)
                hh = meta.get('header_hierarchy') or []
                if hh:
                    header_hiers.append(hh)
        except Exception:
            pass
        return table_ids, header_hiers
    
    def _chunk_content(self, content):
        """
        Splits the document content into chunks based on the specified format and criteria.
        
        Yields:
            tuple: A tuple containing the chunked content and the number of tokens in the chunk.
        """
        #content, placeholders, tables = self._replace_html_tables(content)

        #content = self._markdown_to_plain_text(content)
        #splitter = self._choose_splitter()

        #chunks = splitter.split_text(content)
        chunks = self._splitMarkdownDocument(content)
        #chunks = self._restore_original_tables(chunks, placeholders, tables)

        for chunked_content in chunks:
            chunk_size = self.token_estimator.estimate_tokens(chunked_content["content"])
            if chunk_size > self.max_chunk_size:
                logging.info(f"[doc_analysis_chunker][{self.filename}] truncating {chunk_size} size chunk to fit within {self.max_chunk_size} tokens")
                chunked_content["content"] = self._truncate_chunk(chunked_content["content"])

            yield chunked_content["content"], chunked_content["headers"], chunk_size

    def _splitMarkdownDocument(self, content):
        headers_to_split_on = [
        ("#", "1"),
        ("##", "2"),
        ("###", "3"),
        ("####", "4"),
        ("#####", "5"),
        ("######", "6"),
        ("#######", "7"),
        ("########", "8"),
        ]
        text_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            return_each_line=False,
            strip_headers=False
        )
        character_splitter: RecursiveCharacterTextSplitter
        try:
            character_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.token_overlap
            )
        except Exception as e:
            logging.exception("Failed to load text splitter")
        # Split markdown content into chunks based on headers
        md_chunks = text_splitter.split_text(content)
        # Further split the markdown chunks into the desired
        #char_chunks = character_splitter.split_documents(md_chunks)
        # Return chunk content and headers
        chunks = [
            {
                "content": document.page_content,
                "headers": [
                    document.metadata[header]
                    for header in sorted(document.metadata.keys())
                ],
            }
            for document in md_chunks
        ]
        return chunks

    def _replace_html_tables(self, content):
        """
        Replaces HTML tables in the content with placeholders.
        
        Args:
            content (str): The document content.
        
        Returns:
            tuple: The content with placeholders and a list of the original tables.
        """
        table_pattern = r"(<table[\s\S]*?</table>)"
        tables = re.findall(table_pattern, content, re.IGNORECASE)
        placeholders = [f"TABLE___{i}" for i in range(len(tables))]
        for placeholder, table in zip(placeholders, tables):
            content = content.replace(table, placeholder)
        return content, placeholders, tables
    
    def _markdown_to_plain_text(self, content):
        """
        Converts Markdown content to HTML.
        
        Args:
            content (str): The document content.
        
        Returns:
            str: Content converted to HTML.
        """
        html_content = markdown.markdown(content)
        soup = BeautifulSoup(html_content, 'html.parser')
        plain_text = soup.get_text(separator='\n')
        return plain_text

    def _restore_original_tables(self, chunks, placeholders, tables):
        """
        Restores original tables in the chunks from placeholders.
        
        Args:
            chunks (list): The list of text chunks.
            placeholders (list): The list of table placeholders.
            tables (list): The list of original tables.
        
        Returns:
            list: The list of chunks with original tables restored.
        """
        for placeholder, table in zip(placeholders, tables):
            chunks = [chunk.replace(placeholder, table) for chunk in chunks]
        return chunks

    def _choose_splitter(self):
        """
        Chooses the appropriate splitter based on document format.
        
        Returns:
            object: The splitter to use for chunking.
        """
        separators = [".", "!", "?"] + ['\n', '\t', '}', '{', ']', '[', ')', '(', ' ', ':', ';', ',']
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            separators=separators,
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.token_overlap
        )

    def _number_pagebreaks(self, content):
        """
        Finds and numbers all PageBreaks in the content.
        
        Args:
            content (str): The document content.
        
        Returns:
            str: Content with numbered PageBreaks.
        """
        pagebreaks = re.findall(r'<!-- PageBreak -->', content)
        for i, _ in enumerate(pagebreaks, 1):
            content = content.replace('<!-- PageBreak -->', f'<!-- PageBreak{str(i).zfill(5)} -->', 1)
        return content

    def _update_page(self, content, current_page):
        """
        Updates the current page number based on the content.
        
        Args:
            content (str): The content chunk being processed.
            current_page (int): The current page number.
        
        Returns:
            int: The updated current page number.
        """
        matches = re.findall(r'PageBreak(\d{5})', content)
        if matches:
            page_number = int(matches[-1])
            if page_number >= current_page:
                current_page = page_number + 1
        return current_page

    def _determine_chunk_page(self, content, current_page):
        """
        Determines the chunk page number based on the position of the PageBreak element.
        
        Args:
            content (str): The content chunk being processed.
            current_page (int): The current page number.
        
        Returns:
            int: The page number for the chunk.
        """
        match = re.search(r'PageBreak(\d{5})', content)
        if match:
            page_number = int(match.group(1))
            position = match.start() / len(content)
            # Determine the chunk_page based on the position of the PageBreak element
            if position < 0.5:
                chunk_page = page_number + 1
            else:
                chunk_page = page_number
        else:
            chunk_page = current_page
        return chunk_page

    def _truncate_chunk(self, text):
        """
        Truncates and normalizes the text to ensure it fits within the maximum chunk size.
        
        This method first cleans up the text by removing unnecessary spaces and line breaks. 
        If the text still exceeds the maximum token limit, it iteratively truncates the text 
        until it fits within the limit.

        This method overrides the parent class's method because it includes logic to retain 
        PageBreaks within the truncated text.
        
        Args:
            text (str): The text to be truncated and normalized.
        
        Returns:
            str: The truncated and normalized text.
        """
        # Clean up text (e.g. line breaks)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[\n\r]+', ' ', text).strip()

        page_breaks = re.findall(r'PageBreak\d{5}', text)

        # Truncate if necessary
        if self.token_estimator.estimate_tokens(text) > self.max_chunk_size:
            logging.info(f"[doc_analysis_chunker][{self.filename}] token limit reached, truncating...")
            step_size = 1  # Initial step size
            iteration = 0  # Iteration counter

            while self.token_estimator.estimate_tokens(text) > self.max_chunk_size:
                # Truncate the text
                text = text[:-step_size]
                iteration += 1

                # Increase step size exponentially every 5 iterations
                if iteration % 5 == 0:
                    step_size = min(step_size * 2, 100)

        # Reinsert page breaks and recheck size
        for page_break in page_breaks:
            page_break_text = f" <!-- {page_break} -->"
            if page_break not in text:
                # Calculate the size needed for the page break addition
                needed_size = self.token_estimator.estimate_tokens(page_break_text)

                # Truncate exactly the size needed to accommodate the page break
                while self.token_estimator.estimate_tokens(text) + needed_size > self.max_chunk_size:
                    text = text[:-1]  # Remove one character at a time

                # Now add the page break
                text += page_break_text

        return text
