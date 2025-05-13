import requests
import json
import os
import subprocess
import tempfile
import datetime
from dotenv import load_dotenv
from google import genai
from google.generativeai import types

load_dotenv()

class SpeechToText:
    def __init__(self):
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if not self.elevenlabs_api_key:
            raise ValueError("ELEVENLABS_API_KEY não configurada nas variáveis de ambiente.")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY não configurada nas variáveis de ambiente.")

        genai.configure(api_key=self.gemini_api_key)
        
        self.gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.gemini_client = genai.GenerativeModel(self.gemini_model_name)
        
        self.elevenlabs_scribe_model_id = os.getenv("ELEVENLABS_SCRIBE_MODEL_ID", "scribe_v1")
        
        self.MAX_AUDIO_SIZE_BYTES = int(os.getenv('MAX_AUDIO_SIZE_BYTES', 20 * 1024 * 1024)) # Mantido, mas a ElevenLabs tem seus próprios limites (1GB para arquivo, 2GB para URL)

    def get_file_size(self, file_path):
        return os.path.getsize(file_path)

    def get_audio_duration(self, audio_file_path):
        result = subprocess.run(['ffprobe', '-i', audio_file_path, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=%s' % ("p=0")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return float(result.stdout)

    def resize_audio_if_needed(self, audio_file_path):
        # Esta função pode precisar de ajustes ou ser removida dependendo dos limites e tratamento da ElevenLabs
        # Por enquanto, vamos mantê-la, mas ciente que a ElevenLabs lida com arquivos grandes.
        audio_size = self.get_file_size(audio_file_path)
        if audio_size > self.MAX_AUDIO_SIZE_BYTES: # Comparando com um limite antigo, ElevenLabs suporta mais
            print(f"Aviso: O áudio ({audio_size} bytes) excede MAX_AUDIO_SIZE_BYTES ({self.MAX_AUDIO_SIZE_BYTES} bytes) definido localmente, mas a ElevenLabs pode suportá-lo.")
            # Não vamos redimensionar automaticamente por enquanto, pois a ElevenLabs tem limites maiores.
            # Se o envio falhar devido ao tamanho, esta lógica pode ser reativada ou ajustada.
            # current_duration = self.get_audio_duration(audio_file_path)
            # target_duration = current_duration * self.MAX_AUDIO_SIZE_BYTES / audio_size
            # temp_dir = tempfile.mkdtemp()
            # print(f"Compressed audio will be stored in {temp_dir}")
            # compressed_audio_path = os.path.join(temp_dir, f'compressed_audio_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.wav')
            # subprocess.run(['ffmpeg', '-i', audio_file_path, '-ss', '0', '-t', str(target_duration), compressed_audio_path])
            # return compressed_audio_path
        return audio_file_path

    def transcribe_audio(self, audio_file_path):
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        headers = {
            "xi-api-key": self.elevenlabs_api_key
        }
        data = {
            "model_id": self.elevenlabs_scribe_model_id
            # Outros parâmetros como language_code, num_speakers podem ser adicionados aqui se necessário
        }
        
        with open(audio_file_path, 'rb') as audio_file:
            files = {
                'file': (os.path.basename(audio_file_path), audio_file, 'audio/mpeg') # Tentar inferir ou usar um tipo comum
            }
            try:
                response = requests.post(url, headers=headers, data=data, files=files)
                response.raise_for_status()  # Levanta um erro para códigos de status HTTP ruins (4xx ou 5xx)
                transcript_data = response.json()
                print("Transcribe (ElevenLabs): Done")
                # A documentação indica que o texto completo está em transcript_data["text"]
                return transcript_data.get("text", "") 
            except requests.exceptions.RequestException as e:
                print(f"Erro na transcrição com ElevenLabs: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Detalhes do erro: {e.response.text}")
                return None # Retornar None ou levantar uma exceção mais específica
            except json.JSONDecodeError:
                print(f"Erro ao decodificar JSON da resposta da ElevenLabs: {response.text}")
                return None

    def _generate_gemini_content(self, system_prompt, user_transcription):
        prompt_parts = [
            types.Part.from_text(f"{system_prompt}\n\nTexto para analisar:\n{user_transcription}")
        ]
        contents = [types.Content(role="user", parts=prompt_parts)]
        
        try:
            response = self.gemini_client.generate_content(contents=contents)
            return response.text
        except Exception as e:
            print(f"Erro ao gerar conteúdo com Gemini: {e}")
            # Você pode querer verificar response.prompt_feedback aqui também
            # if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            #     print(f"Prompt Feedback: {response.prompt_feedback}")
            return f"Erro ao gerar conteúdo: {e}"


    def abstract_summary_extraction(self, transcription):
        system_prompt = "Você é uma IA altamente qualificada, treinada em compreensão de linguagem e sumarização. Gostaria que você lesse o texto a seguir e o resumisse em um parágrafo abstrato conciso. Procure reter os pontos mais importantes, fornecendo um resumo coerente e legível que possa ajudar uma pessoa a entender os pontos principais da discussão sem precisar ler o texto inteiro. Evite detalhes desnecessários ou pontos tangenciais."
        summary = self._generate_gemini_content(system_prompt, transcription)
        print("Summary (Gemini): Done")
        return summary

    def key_points_extraction(self, transcription):
        system_prompt = "Você é uma IA proficiente com especialidade em destilar informações em pontos-chave. Com base no texto a seguir, identifique e liste os principais pontos que foram discutidos ou levantados. Devem ser as ideias, descobertas ou tópicos mais importantes que são cruciais para a essência da discussão. Seu objetivo é fornecer uma lista que alguém possa ler para entender rapidamente sobre o que foi falado."
        key_points = self._generate_gemini_content(system_prompt, transcription)
        print("Key Points (Gemini): Done")
        return key_points

    def action_item_extraction(self, transcription):
        system_prompt = "Você é um especialista em IA em analisar conversas e extrair itens de ação. Revise o texto e identifique quaisquer tarefas, atribuições ou ações que foram acordadas ou mencionadas como necessitando ser feitas. Podem ser tarefas atribuídas a indivíduos específicos ou ações gerais que o grupo decidiu tomar. Liste esses itens de ação de forma clara e concisa."
        action_items = self._generate_gemini_content(system_prompt, transcription)
        print("Action Items (Gemini): Done")
        return action_items

    def sentiment_analysis(self, transcription):
        system_prompt = "Como uma IA com experiência em análise de linguagem e emoção, sua tarefa é analisar o sentimento do texto a seguir. Considere o tom geral da discussão, a emoção transmitida pela linguagem utilizada e o contexto em que palavras e frases são usadas. Indique se o sentimento é geralmente positivo, negativo ou neutro e forneça breves explicações para sua análise, quando possível."
        sentiment = self._generate_gemini_content(system_prompt, transcription)
        print("Sentiment (Gemini): Done")
        return sentiment

    def meeting_minutes(self, transcription):
        abstract_summary = self.abstract_summary_extraction(transcription)
        key_points = self.key_points_extraction(transcription)
        action_items = self.action_item_extraction(transcription)
        sentiment = self.sentiment_analysis(transcription)
        return {
            'abstract_summary': abstract_summary,
            'key_points': key_points,
            'action_items': action_items,
            'sentiment': sentiment
        }

    def store_in_json_file(self, data):
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, f'meeting_data_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.json')
        print(f"JSON file path: {file_path}")
        with open(file_path, 'w') as f:
            json.dump(data, f)
        print("JSON file created successfully.")

    def transcribe(self, audio_file_path):
        audio_file_path = self.resize_audio_if_needed(audio_file_path)
        transcription = self.transcribe_audio(audio_file_path)
        summary = self.meeting_minutes(transcription)
        self.store_in_json_file(summary)
    
        print(f"Abstract Summary: {summary['abstract_summary']}")
        print(f"Key Points: {summary['key_points']}")
        print(f"Action Items: {summary['action_items']}")
        print(f"Sentiment: {summary['sentiment']}")

