import re
import numpy as np
import pandas as pd
from typing import Optional, Union, Literal, List, Tuple, Dict, Any, Callable
from datetime import datetime
import json
import pickle
from collections import deque, defaultdict
import requests
import os

class IntelligenceArtificial:
    """
    Clase especializada en IA clásica: agentes, chatbots, sistemas expertos,
    razonamiento simbólico, búsqueda, optimización y conexión a modelos de IA.
    """
    
    def __init__(self, backend_ia: Tuple[Literal['mistral', 'deepseek'], str] = ('mistral', '')):
        """
        Inicializa el sistema de IA.
        
        Args:
            backend_ia: Tupla con (nombre_backend, api_key). 
                       Solo soporta 'mistral' o 'deepseek' a través de OpenRouter.
                       Si la api_key está vacía, se lee de la variable de
                       entorno OPENROUTER_API_KEY.
        """
        # Configuración de IA - Solo Mistral o DeepSeek
        self.backend_ia = backend_ia[0]
        self.api_key = (
            backend_ia[1] if len(backend_ia) > 1 and backend_ia[1] else ''
        ) or os.environ.get('OPENROUTER_API_KEY', '')
        
        # Verificar que el backend sea válido
        if self.backend_ia not in ['mistral', 'deepseek']:
            raise ValueError(f"Backend '{self.backend_ia}' no soportado. Use 'mistral' o 'deepseek'.")
        
        # Sistemas de conocimiento
        self._knowledge_base = defaultdict(list)  # Base de conocimiento
        self._working_memory = {}  # Memoria de trabajo
        self._production_rules = []  # Reglas de producción
        self._inference_engine = None
        
        # Agentes
        self._agents = {}
        self._agent_states = {}
        
        # Chatbot
        self._intents = {}
        self._dialog_states = {}
        self._conversation_history = []
        
        # Configuración de APIs - SOLO OpenRouter
        self._api_endpoint = 'https://openrouter.ai/api/v1/chat/completions'  # Único endpoint
        
        # Modelos disponibles para cada backend
        self._available_models = {
            'mistral': {
                'mistral-small': 'mistralai/mistral-small-3.1-24b-instruct:free',
                'mistral-medium': 'mistralai/mistral-medium',
                'mistral-large': 'mistralai/mistral-large-latest'
            },
            'deepseek': {
                'deepseek-r1': 'deepseek/deepseek-r1-0528:free',
                'deepseek-chat': 'deepseek/deepseek-chat',
                'deepseek-coder': 'deepseek/deepseek-coder'
            }
        }
        
        # Modelo por defecto para cada backend
        self._default_models = {
            'mistral': 'mistralai/mistral-small-3.1-24b-instruct:free',
            'deepseek': 'deepseek/deepseek-r1-0528:free'
        }

    # =========================
    # SISTEMA DE CONOCIMIENTO
    # =========================
    
    def add_fact(self, subject: str, predicate: str, value: Any):
        """
        Agrega un hecho a la base de conocimiento.
        
        Ejemplo:
            add_fact('perro', 'es', 'animal')
            add_fact('fido', 'es', 'perro')
        """
        self._knowledge_base[subject].append((predicate, value))
        return f"Hecho agregado: {subject} {predicate} {value}"
    
    def query_fact(self, subject: str, predicate: str = None) -> List[Any]:
        """
        Consulta hechos en la base de conocimiento.
        
        Args:
            subject: Sujeto a consultar
            predicate: Predicado específico (opcional)
        
        Returns:
            Lista de valores que cumplen la consulta
        """
        if subject not in self._knowledge_base:
            return []
        
        if predicate:
            return [value for (p, value) in self._knowledge_base[subject] if p == predicate]
        else:
            return [value for (_, value) in self._knowledge_base[subject]]
    
    def add_rule(self, name: str, condition: Callable, action: Callable, priority: int = 1):
        """
        Agrega una regla de producción al sistema experto.
        
        Args:
            name: Nombre de la regla
            condition: Función que evalúa si la regla se activa
            action: Función que se ejecuta si la condición es verdadera
            priority: Prioridad de la regla (mayor = más prioritario)
        """
        self._production_rules.append({
            'name': name,
            'condition': condition,
            'action': action,
            'priority': priority
        })
        # Ordenar por prioridad
        self._production_rules.sort(key=lambda x: x['priority'], reverse=True)
        return f"Regla '{name}' agregada con prioridad {priority}"
    
    def inference_engine(self, context: Dict[str, Any]) -> List[str]:
        """
        Motor de inferencia que evalúa reglas y ejecuta acciones.
        
        Args:
            context: Contexto actual para evaluar condiciones
        
        Returns:
            Lista de acciones ejecutadas
        """
        executed_actions = []
        
        for rule in self._production_rules:
            try:
                if rule['condition'](context):
                    result = rule['action'](context)
                    executed_actions.append(f"{rule['name']}: {result}")
                    # Actualizar contexto con el resultado
                    context.update(result if isinstance(result, dict) else {})
            except Exception as e:
                executed_actions.append(f"Error en regla {rule['name']}: {str(e)}")
        
        return executed_actions

    # =========================
    # BÚSQUEDA Y OPTIMIZACIÓN
    # =========================
    
    def bfs(self, start, goal, graph: dict):
        """Búsqueda en amplitud."""
        visited = set()
        queue = deque([[start]])
        
        while queue:
            path = queue.popleft()
            node = path[-1]
            
            if node == goal:
                return path
            
            if node not in visited:
                visited.add(node)
                for neighbor in graph.get(node, []):
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)
        
        return None
    
    def dfs(self, start, goal, graph: dict, depth_limit: int = 100):
        """Búsqueda en profundidad."""
        visited = set()
        
        def dfs_recursive(current, path, depth):
            if depth > depth_limit:
                return None
            
            if current == goal:
                return path
            
            if current in visited:
                return None
            
            visited.add(current)
            
            for neighbor in graph.get(current, []):
                new_path = path + [neighbor]
                result = dfs_recursive(neighbor, new_path, depth + 1)
                if result:
                    return result
            
            return None
        
        return dfs_recursive(start, [start], 0)
    
    def a_star(self, start, goal, graph: dict, heuristic_fn: Callable = None):
        """Algoritmo A*."""
        if heuristic_fn is None:
            heuristic_fn = lambda x, y: 0
        
        open_set = {start}
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic_fn(start, goal)}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            open_set.remove(current)
            
            for neighbor in graph.get(current, []):
                tentative_g_score = g_score[current] + 1
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic_fn(neighbor, goal)
                    if neighbor not in open_set:
                        open_set.add(neighbor)
        
        return None
    
    def genetic_algorithm(
        self,
        fitness_fn: Callable,
        population_size: int = 10,
        generations: int = 20,
        gene_length: int = 10
    ):
        """Algoritmo genético básico."""
        # Generar población inicial
        population = [np.random.randint(0, 2, gene_length) for _ in range(population_size)]
        best_individual = None
        best_fitness = -float('inf')
        
        for gen in range(generations):
            # Evaluar fitness
            fitness_scores = [fitness_fn(ind) for ind in population]
            
            # Encontrar el mejor
            current_best_idx = np.argmax(fitness_scores)
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_individual = population[current_best_idx]
            
            # Selección (ruleta)
            fitness_sum = sum(fitness_scores)
            probs = [score/fitness_sum for score in fitness_scores]
            selected_indices = np.random.choice(
                range(population_size),
                size=population_size,
                p=probs
            )
            
            # Cruzamiento y mutación
            new_population = []
            for i in range(0, population_size, 2):
                parent1 = population[selected_indices[i]]
                parent2 = population[selected_indices[i+1]]
                
                # Cruzamiento
                crossover_point = np.random.randint(1, gene_length-1)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                
                # Mutación
                mutation_rate = 0.1
                for j in range(gene_length):
                    if np.random.random() < mutation_rate:
                        child1[j] = 1 - child1[j]
                    if np.random.random() < mutation_rate:
                        child2[j] = 1 - child2[j]
                
                new_population.extend([child1, child2])
            
            population = new_population[:population_size]
        
        return best_individual, best_fitness
    
    def simulated_annealing(
        self,
        objective_fn: Callable,
        initial_solution: Any = None,
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        iterations: int = 1000
    ):
        """Recocido simulado."""
        if initial_solution is None:
            current_solution = np.random.random(10)  # Solución por defecto
        else:
            current_solution = initial_solution
        
        current_energy = objective_fn(current_solution)
        best_solution = current_solution.copy()
        best_energy = current_energy
        
        temp = initial_temp
        
        for i in range(iterations):
            # Generar vecino
            neighbor = current_solution + np.random.normal(0, 0.1, len(current_solution))
            neighbor_energy = objective_fn(neighbor)
            
            # Decidir si aceptar
            delta_energy = neighbor_energy - current_energy
            
            if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temp):
                current_solution = neighbor
                current_energy = neighbor_energy
                
                if current_energy < best_energy:
                    best_solution = current_solution.copy()
                    best_energy = current_energy
            
            # Enfriar
            temp *= cooling_rate
        
        return best_solution, best_energy

    # =========================
    # AGENTES INTELIGENTES
    # =========================
    
    def create_agent(self, agent_id: str, agent_type: Literal['simple', 'goal', 'utility'] = 'simple', **kwargs):
        """
        Crea un agente inteligente.
        
        Args:
            agent_id: Identificador único del agente
            agent_type: Tipo de agente
            **kwargs: Parámetros específicos del tipo de agente
        
        Returns:
            Función del agente creado
        """
        if agent_type == 'simple':
            agent = self._create_simple_agent(**kwargs)
        elif agent_type == 'goal':
            agent = self._create_goal_based_agent(**kwargs)
        elif agent_type == 'utility':
            agent = self._create_utility_agent(**kwargs)
        else:
            raise ValueError(f"Tipo de agente no soportado: {agent_type}")
        
        self._agents[agent_id] = agent
        self._agent_states[agent_id] = {'status': 'active', 'history': []}
        
        return agent
    
    def _create_simple_agent(self, perceive_fn: Callable, act_fn: Callable):
        """Crea un agente simple basado en percepción-acción."""
        def agent():
            perception = perceive_fn()
            action = act_fn(perception)
            return action
        return agent
    
    def _create_goal_based_agent(self, goal: Any, actions: List[Callable]):
        """Crea un agente basado en objetivos."""
        def agent(current_state):
            for action in actions:
                new_state = action(current_state)
                if self._is_goal_achieved(new_state, goal):
                    return {'action': action.__name__, 'state': new_state, 'goal_achieved': True}
            return {'action': 'no_action', 'state': current_state, 'goal_achieved': False}
        return agent
    
    def _create_utility_agent(self, utility_fn: Callable, actions: List[Callable]):
        """Crea un agente basado en utilidad."""
        def agent(current_state):
            best_action = None
            best_utility = -float('inf')
            
            for action in actions:
                new_state = action(current_state)
                utility = utility_fn(new_state)
                
                if utility > best_utility:
                    best_utility = utility
                    best_action = action
            
            if best_action:
                new_state = best_action(current_state)
                return {
                    'action': best_action.__name__,
                    'state': new_state,
                    'utility': best_utility
                }
            return {'action': 'no_action', 'state': current_state, 'utility': 0}
        return agent
    
    def _is_goal_achieved(self, state: Any, goal: Any) -> bool:
        """Verifica si se ha alcanzado un objetivo."""
        if callable(goal):
            return goal(state)
        return state == goal
    
    def run_agent(self, agent_id: str, initial_state: Any = None, steps: int = 10):
        """
        Ejecuta un agente por un número de pasos.
        
        Args:
            agent_id: ID del agente a ejecutar
            initial_state: Estado inicial
            steps: Número de pasos a ejecutar
        
        Returns:
            Historial de ejecución del agente
        """
        if agent_id not in self._agents:
            return f"Agente '{agent_id}' no encontrado"
        
        agent = self._agents[agent_id]
        history = []
        current_state = initial_state
        
        for step in range(steps):
            try:
                result = agent(current_state) if initial_state is not None else agent()
                history.append({
                    'step': step,
                    'state': current_state,
                    'result': result
                })
                
                if isinstance(result, dict) and 'state' in result:
                    current_state = result['state']
                
                # Verificar si se alcanzó un objetivo
                if isinstance(result, dict) and result.get('goal_achieved', False):
                    break
                    
            except Exception as e:
                history.append({
                    'step': step,
                    'error': str(e)
                })
                break
        
        self._agent_states[agent_id]['history'].extend(history)
        return history

    # =========================
    # CHATBOT INTELIGENTE
    # =========================
    
    def add_intent(self, name: str, patterns: list, response: str):
        """Agrega un intent para el chatbot."""
        self._intents[name] = {
            'patterns': patterns,
            'response': response
        }
        return f"Intent '{name}' agregado con {len(patterns)} patrones."
    
    def remove_intent(self, name: str):
        """Elimina un intent."""
        if name in self._intents:
            del self._intents[name]
            return f"Intent '{name}' eliminado."
        return f"Intent '{name}' no encontrado."
    
    def _match_intent(self, text: str) -> Optional[str]:
        """Busca coincidencia con intents existentes."""
        text_lower = text.lower()
        for intent_name, intent_data in self._intents.items():
            for pattern in intent_data['patterns']:
                if pattern.lower() in text_lower:
                    return intent_name
        return None
    
    def create_chatbot(self, chatbot_id: str, personality: str = "helpful assistant"):
        """
        Crea un chatbot con personalidad específica.
        
        Args:
            chatbot_id: ID único del chatbot
            personality: Descripción de la personalidad
        
        Returns:
            Función del chatbot
        """
        self._dialog_states[chatbot_id] = {
            'personality': personality,
            'context': {},
            'memory': []
        }
        
        def chatbot_handler(message: str, use_backend: bool = True):
            return self._process_chat_message(chatbot_id, message, use_backend)
        
        return chatbot_handler
    
    def _process_chat_message(self, chatbot_id: str, message: str, use_backend: bool = True) -> str:
        """Procesa un mensaje del chatbot."""
        # Agregar al historial
        self._conversation_history.append({
            'chatbot': chatbot_id,
            'role': 'user',
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
        state = self._dialog_states[chatbot_id]
        
        # Primero intentar razonamiento simbólico
        symbolic_response = self._symbolic_reasoning(message, state['context'])
        if symbolic_response and symbolic_response != message:
            response = symbolic_response
        elif use_backend and self.backend_ia != 'local':
            # Usar backend de IA
            response = self._call_ai_api(message, state['personality'])
        else:
            # Respuesta por defecto basada en reglas
            response = self._rule_based_response(message, state)
        
        # Guardar en memoria del chatbot
        state['memory'].append({
            'input': message,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Agregar al historial
        self._conversation_history.append({
            'chatbot': chatbot_id,
            'role': 'assistant',
            'message': response,
            'timestamp': datetime.now().isoformat()
        })
        
        return response
    
    def _symbolic_reasoning(self, message: str, context: dict) -> Optional[str]:
        """Razonamiento simbólico sobre el mensaje."""
        message_lower = message.lower()
        
        # Consultar base de conocimiento
        if 'qué es' in message_lower:
            # Extraer concepto
            words = message_lower.split()
            for i, word in enumerate(words):
                if word == 'es' and i > 0:
                    concept = words[i-1]
                    facts = self.query_fact(concept)
                    if facts:
                        return f"{concept} es {', '.join(map(str, facts))}"
        
        # Inferencia basada en reglas
        if 'si' in message_lower and 'entonces' in message_lower:
            return "Parece que estás definiendo una regla lógica. ¿Quieres que la agregue al sistema?"
        
        return None
    
    def _rule_based_response(self, message: str, state: dict) -> str:
        """Genera respuesta basada en reglas predefinidas."""
        message_lower = message.lower()
        
        # Primero verificar intents personalizados
        matched_intent = self._match_intent(message)
        if matched_intent:
            return self._intents[matched_intent]['response']
        
        # Reglas de diálogo por defecto
        rules = {
            r'hola|buenos días|buenas tardes': f"¡Hola! Soy un asistente con personalidad {state['personality']}. ¿En qué puedo ayudarte?",
            r'cómo estás|qué tal': "¡Estoy funcionando perfectamente! Listo para ayudarte con lo que necesites.",
            r'gracias|muchas gracias': "¡De nada! Es un placer ayudarte.",
            r'adiós|hasta luego|chao': "¡Hasta luego! Que tengas un excelente día.",
            r'qué puedes hacer|qué sabes hacer': "Puedo ayudarte con razonamiento lógico, responder preguntas, ejecutar agentes inteligentes y conectarme con modelos de IA avanzados.",
            r'cuál es tu nombre|quién eres': f"Soy un agente de IA con personalidad: {state['personality']}",
            r'ayuda|help': self._get_help_text(),
            r'explica.*ia|qué es.*inteligencia artificial': "La Inteligencia Artificial es la simulación de procesos de inteligencia humana por máquinas, especialmente sistemas informáticos. Incluye aprendizaje, razonamiento, autocorrección y más."
        }
        
        for pattern, response in rules.items():
            if re.search(pattern, message_lower, re.IGNORECASE):
                return response
        
        # Respuesta por defecto
        return "Interesante pregunta. ¿Te gustaría que consulte con un modelo de IA más avanzado para darte una mejor respuesta? (Escribe 'sí' para continuar)"
    
    def chat_stream(self, text: str):
        """Simula stream de chat (implementación básica)."""
        response = self._process_chat_message('default', text)
        for char in response:
            yield char
    
    def reset_conversation(self):
        """Reinicia el historial de conversación."""
        self._conversation_history = []
        return "Conversación reiniciada."

    # =========================
    # CONEXIÓN CON MODELOS DE IA MODERNOS
    # =========================
    
    def _get_model(self, model_name: str = None) -> str:
        """
        Obtiene el modelo correcto para el backend actual.
        
        Args:
            model_name: Nombre corto del modelo (opcional)
        
        Returns:
            Modelo completo para OpenRouter API
        """
        if model_name:
            # Si se especifica un modelo, buscar en los disponibles
            if model_name in self._available_models[self.backend_ia]:
                return self._available_models[self.backend_ia][model_name]
            else:
                # Si no está en la lista, usar como está (podría ser un modelo completo)
                return model_name
        else:
            # Usar modelo por defecto del backend
            return self._default_models[self.backend_ia]
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Obtiene los modelos disponibles para cada backend.
        
        Returns:
            Diccionario con modelos disponibles
        """
        return {
            'backend': self.backend_ia,
            'models': list(self._available_models[self.backend_ia].keys()),
            'default': list(self._default_models.keys())[0] if self.backend_ia == 'mistral' else list(self._default_models.keys())[1]
        }
    
    def add_model(self, model_key: str, model_id: str):
        """
        Agrega un modelo personalizado al backend actual.
        
        Args:
            model_key: Clave para identificar el modelo (ej: 'mi-modelo')
            model_id: ID completo del modelo en OpenRouter (ej: 'organizacion/nombre-modelo')
        
        Returns:
            Mensaje de confirmación
        """
        if model_key in self._available_models[self.backend_ia]:
            return f"El modelo '{model_key}' ya existe. Usa update_model para actualizarlo."
        
        self._available_models[self.backend_ia][model_key] = model_id
        return f"Modelo '{model_key}' agregado al backend {self.backend_ia}"
    
    def update_model(self, model_key: str, model_id: str):
        """
        Actualiza un modelo existente.
        
        Args:
            model_key: Clave del modelo a actualizar
            model_id: Nuevo ID del modelo
        
        Returns:
            Mensaje de confirmación
        """
        if model_key not in self._available_models[self.backend_ia]:
            return f"El modelo '{model_key}' no existe. Usa add_model para agregarlo."
        
        self._available_models[self.backend_ia][model_key] = model_id
        return f"Modelo '{model_key}' actualizado en backend {self.backend_ia}"
    
    def _call_ai_api(self, message: str, personality: str, model: str = None, messages: list = None) -> str:
        """Realiza llamada a la API de IA a través de OpenRouter."""
        if not self.api_key:
            return "API key no configurada. Configura tu API key en el constructor."
        
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            # Obtener el modelo correcto
            model_to_use = self._get_model(model)
            
            # Preparar mensajes si no se proporcionan
            if messages is None:
                messages = [
                    {"role": "system", "content": f"Eres un asistente con la siguiente personalidad: {personality}"},
                    {"role": "user", "content": message}
                ]
            
            # Construir payload según OpenRouter API
            payload = {
                'model': model_to_use,
                'messages': messages,
                'temperature': 0.7,
                'max_tokens': 1000
            }
            
            # Llamar al endpoint único de OpenRouter
            response = requests.post(
                self._api_endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                return response_data['choices'][0]['message']['content']
            else:
                error_msg = f"Error en la API de OpenRouter ({response.status_code}): "
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg += error_data['error'].get('message', str(error_data))
                    else:
                        error_msg += str(error_data)
                except:
                    error_msg += str(response.text)
                return error_msg
                
        except requests.exceptions.Timeout:
            return "Error: Timeout al conectar con la API de OpenRouter. Intenta nuevamente."
        except requests.exceptions.ConnectionError:
            return "Error: No se pudo conectar con OpenRouter. Verifica tu conexión a internet."
        except Exception as e:
            return f"Error al conectar con OpenRouter ({self.backend_ia}): {str(e)}"
    
    def chat_with_ai(self, message: str, model: str = None, personality: str = "helpful assistant") -> str:
        """
        Chatea con modelos de IA modernos a través de OpenRouter.
        
        Args:
            message: Mensaje del usuario
            model: Modelo a usar (opcional, usa el modelo por defecto del backend)
                  Para Mistral: 'mistral-small', 'mistral-medium', 'mistral-large'
                  Para DeepSeek: 'deepseek-r1', 'deepseek-chat', 'deepseek-coder'
            personality: Personalidad del asistente
        
        Returns:
            Respuesta del modelo de IA
        """
        if not self.api_key:
            return "API key no configurada. Configura tu API key en el constructor."
        
        # Validar que el modelo sea compatible con el backend
        if model and self.backend_ia == 'mistral' and not model.startswith('mistral'):
            print(f"Advertencia: Modelo '{model}' puede no ser compatible con backend Mistral")
        elif model and self.backend_ia == 'deepseek' and not model.startswith('deepseek'):
            print(f"Advertencia: Modelo '{model}' puede no ser compatible con backend DeepSeek")
        
        # Preparar historial de conversación
        messages = []
        
        # Agregar historial reciente (últimos 6 mensajes)
        for msg in self._conversation_history[-6:]:
            messages.append({
                'role': msg['role'],
                'content': msg['message'] if 'message' in msg else msg['content']
            })
        
        # Agregar mensaje actual
        messages.append({'role': 'user', 'content': message})
        
        # Llamar a la API de OpenRouter
        response = self._call_ai_api(message, personality, model, messages)
        
        # Guardar en historial
        self._conversation_history.append({
            'role': 'user',
            'message': message,
            'backend': self.backend_ia,
            'model': model if model else 'default',
            'timestamp': datetime.now().isoformat()
        })
        
        self._conversation_history.append({
            'role': 'assistant',
            'message': response,
            'backend': self.backend_ia,
            'model': model if model else 'default',
            'timestamp': datetime.now().isoformat()
        })
        
        return response

    # =========================
    # RAZONAMIENTO Y BÚSQUEDA SIMBÓLICA
    # =========================
    
    def forward_chaining(self, initial_facts: List[str], max_iterations: int = 100) -> Dict[str, Any]:
        """
        Encadenamiento hacia adelante.
        
        Args:
            initial_facts: Hechos iniciales
            max_iterations: Máximo de iteraciones
        
        Returns:
            Todos los hechos derivados
        """
        facts = set(initial_facts)
        new_facts_added = True
        iteration = 0
        
        while new_facts_added and iteration < max_iterations:
            new_facts_added = False
            iteration += 1
            
            for rule in self._production_rules:
                try:
                    # Simular evaluación de condición
                    context = {'facts': list(facts)}
                    if rule['condition'](context):
                        result = rule['action'](context)
                        if isinstance(result, str) and result not in facts:
                            facts.add(result)
                            new_facts_added = True
                except:
                    continue
        
        return {
            'iterations': iteration,
            'total_facts': len(facts),
            'facts': list(facts),
            'rules_applied': iteration - 1
        }
    
    def backward_chaining(self, goal: str, known_facts: List[str]) -> Tuple[bool, List[str]]:
        """
        Encadenamiento hacia atrás.
        
        Args:
            goal: Objetivo a probar
            known_facts: Hechos conocidos
        
        Returns:
            (es_provable, ruta_de_prueba)
        """
        if goal in known_facts:
            return True, [goal]
        
        # Buscar reglas que puedan probar el objetivo
        supporting_rules = []
        for rule in self._production_rules:
            try:
                # Simular si la acción produce el objetivo
                context = {'facts': known_facts}
                result = rule['action'](context)
                if result == goal:
                    supporting_rules.append(rule)
            except:
                continue
        
        for rule in supporting_rules:
            # Verificar condiciones de la regla
            context = {'facts': known_facts}
            if rule['condition'](context):
                # Extraer subobjetivos (simplificado)
                subgoals = self._extract_subgoals(rule['condition'])
                all_proven = True
                path = [goal]
                
                for subgoal in subgoals:
                    proven, subpath = self.backward_chaining(subgoal, known_facts)
                    if proven:
                        path.extend(subpath)
                    else:
                        all_proven = False
                        break
                
                if all_proven:
                    return True, path
        
        return False, []
    
    def _extract_subgoals(self, condition_fn: Callable) -> List[str]:
        """Extrae subobjetivos de una función de condición."""
        # Esta es una implementación simplificada
        # En un sistema real, necesitarías análisis de la función
        return []

    # =========================
    # HERRAMIENTAS ADICIONALES
    # =========================
    
    def save_state(self, filepath: str):
        """Guarda el estado completo de la IA (maneja funciones lambda)."""
        # Crear una versión serializable del estado
        serializable_state = {
            'knowledge_base': dict(self._knowledge_base),
            'production_rules': [],  # No guardar reglas con lambdas
            'agents': {},  # Los agentes pueden tener funciones también
            'agent_states': self._agent_states,
            'intents': self._intents,
            'dialog_states': self._dialog_states,
            'conversation_history': self._conversation_history,
            'backend_ia': self.backend_ia,
            # La api_key NO se guarda por seguridad; se relee del entorno.
            'available_models': self._available_models,
            'default_models': self._default_models
        }
        
        # Guardar solo las reglas que son serializables (sin lambdas)
        for rule in self._production_rules:
            # Verificar si la regla es serializable
            try:
                # Intentar guardar solo información básica
                serializable_rule = {
                    'name': rule['name'],
                    'priority': rule['priority']
                    # No guardar condition y action si son lambdas
                }
                serializable_state['production_rules'].append(serializable_rule)
            except:
                continue
        
        with open(filepath, 'wb') as f:
            pickle.dump(serializable_state, f)
    
        return f"Estado guardado en {filepath} (nota: funciones lambda no se guardaron)"
    
    def load_state(self, filepath: str):
        """Carga el estado de la IA desde archivo."""
        if not os.path.exists(filepath):
            return f"Archivo no encontrado: {filepath}"
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self._knowledge_base = defaultdict(list, state['knowledge_base'])
        self._production_rules = state['production_rules']
        self._agents = state['agents']
        self._agent_states = state['agent_states']
        self._intents = state['intents']
        self._dialog_states = state['dialog_states']
        self._conversation_history = state['conversation_history']
        self.backend_ia = state['backend_ia']
        self.api_key = state.get('api_key') or os.environ.get('OPENROUTER_API_KEY', '')
        self._available_models = state['available_models']
        self._default_models = state['default_models']
        
        return "Estado cargado exitosamente"
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Obtiene el historial de conversación."""
        return self._conversation_history[-limit:] if limit > 0 else self._conversation_history
    
    def clear_conversation_history(self):
        """Limpia el historial de conversación."""
        self._conversation_history = []
        return "Historial de conversación limpiado"
    
    def _get_help_text(self) -> str:
        """Genera texto de ayuda actualizado."""
        models_info = self.get_available_models()
        
        help_text = f"""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                   🤖 INTELLIGENCE ARTIFICIAL - AYUDA                       ║
    ╚════════════════════════════════════════════════════════════════════════════╝

    📝 DESCRIPCIÓN:
    Sistema de IA clásica con agentes inteligentes, sistemas expertos,
    razonamiento simbólico y conexión a modelos de IA modernos a través de OpenRouter.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ⚙️ CONFIGURACIÓN ACTUAL:
    • Backend: {self.backend_ia}
    • API Endpoint: {self._api_endpoint}
    • Modelos disponibles ({len(models_info['models'])}):
        {', '.join(models_info['models'])}
    • Modelo por defecto: {models_info['default']}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    🔧 MÉTODOS PRINCIPALES:

    📚 SISTEMA DE CONOCIMIENTO:
    • add_fact(subject, predicate, value)    → Agrega hecho a la base de conocimiento
    • query_fact(subject, predicate)         → Consulta hechos
    • add_rule(name, condition, action)      → Agrega regla de producción
    • inference_engine(context)              → Ejecuta motor de inferencia

    🔍 BÚSQUEDA Y OPTIMIZACIÓN:
    • bfs(start, goal, graph)                → Búsqueda en amplitud
    • dfs(start, goal, graph)                → Búsqueda en profundidad
    • a_star(start, goal, graph, heuristic)  → Algoritmo A*
    • genetic_algorithm(fitness_fn)          → Algoritmo genético
    • simulated_annealing(objective_fn)      → Recocido simulado

    🤖 AGENTES INTELIGENTES:
    • create_agent(id, type, **kwargs)       → Crea un agente inteligente
    • run_agent(id, initial_state, steps)    → Ejecuta un agente
    • forward_chaining(facts)                → Encadenamiento hacia adelante
    • backward_chaining(goal, facts)         → Encadenamiento hacia atrás

    💬 CHATBOT Y IA (OpenRouter):
    • create_chatbot(id, personality)        → Crea un chatbot con personalidad
    • add_intent(name, patterns, response)   → Agrega intención al chatbot
    • remove_intent(name)                    → Elimina intención
    • chat_with_ai(message, model, personality) → Chatea con modelos de IA
    • chat_stream(text)                      → Chat en streaming
    • reset_conversation()                   → Reinicia conversación
    • get_conversation_history(limit)        → Obtiene historial de conversación
    • get_available_models()                 → Lista modelos disponibles
    • add_model(key, id)                     → Agrega modelo personalizado
    • update_model(key, id)                  → Actualiza modelo existente

    💾 UTILIDADES:
    • save_state(filepath)                   → Guarda estado completo
    • load_state(filepath)                   → Carga estado guardado
    • clear_conversation_history()           → Limpia historial

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    🚀 EJEMPLOS RÁPIDOS:
    
    1. Sistema experto:
       ia.add_fact('perro', 'es', 'animal')
       ia.add_rule('regla1', lambda ctx: 'perro' in ctx.get('facts', []), 
                   lambda ctx: 'animal encontrado')
    
    2. Chatbot local:
       chatbot = ia.create_chatbot('asistente', 'amigable y útil')
       respuesta = chatbot('Hola, ¿cómo estás?')
    
    3. Chat con IA (Mistral/DeepSeek):
       # Con modelo por defecto
       respuesta = ia.chat_with_ai('Explica la teoría de la relatividad')
       
       # Con modelo específico
       respuesta = ia.chat_with_ai('Escribe un poema', model='mistral-large')
    
    4. Agregar modelo personalizado:
       ia.add_model('mi-modelo', 'organizacion/nombre-modelo:free')
    
    5. Sistema de búsqueda:
       grafo = {{'A': ['B', 'C'], 'B': ['D'], 'C': ['D'], 'D': []}}
       camino = ia.bfs('A', 'D', grafo)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🔗 DOCUMENTACIÓN:
    • OpenRouter API: https://openrouter.ai/docs/api-reference
    • Modelos disponibles: https://openrouter.ai/models
    • Autenticación: https://openrouter.ai/docs/authentication
    """
        return help_text
    
    def help(self):
        """Muestra la ayuda completa."""
        print(self._get_help_text())


# =========================
# EJEMPLOS DE USO COMPLETOS
# =========================

if __name__ == "__main__":
    print("="*60)
    print("🤖 DEMOSTRACIÓN DE INTELLIGENCE ARTIFICIAL")
    print("="*60)
    
    # 1. Crear instancia con Mistral (la API key se lee de OPENROUTER_API_KEY)
    print("\n1. Inicializando con backend Mistral...")
    ia = IntelligenceArtificial(backend_ia=('mistral', ''))
    
    # 2. Mostrar ayuda
    print("\n2. Mostrando ayuda del sistema...")
    ia.help()
    
    # 3. Ver modelos disponibles
    print("\n3. Modelos disponibles para Mistral:")
    modelos = ia.get_available_models()
    print(f"   Backend: {modelos['backend']}")
    print(f"   Modelos: {', '.join(modelos['models'])}")
    
    # 4. Crear un chatbot personalizado
    print("\n4. Creando chatbot personalizado...")
    mi_chatbot = ia.create_chatbot('mi_asistente', 'divertido y creativo')
    
    # 5. Agregar intents personalizados
    print("\n5. Agregando intents personalizados...")
    ia.add_intent('saludo', ['hola', 'buenos días', 'buenas tardes'], 
                  "¡Hola! Soy tu asistente de IA. ¿En qué puedo ayudarte?")
    ia.add_intent('despedida', ['adiós', 'hasta luego', 'chao'], 
                  "¡Hasta luego! Ha sido un placer ayudarte.")
    
    # 6. Chat con intents personalizados
    print("\n6. Probando chatbot con intents...")
    print(f"   Usuario: Hola")
    print(f"   Chatbot: {mi_chatbot('Hola')}")
    
    print(f"   Usuario: Adiós")
    print(f"   Chatbot: {mi_chatbot('Adiós')}")
    
    # 7. Sistema experto simple
    print("\n7. Configurando sistema experto...")
    ia.add_fact('socrates', 'es', 'humano')
    ia.add_fact('todos los humanos', 'son', 'mortales')
    ia.add_rule('mortalidad', 
                lambda ctx: 'humano' in ctx.get('facts', []),
                lambda ctx: {'conclusión': 'mortal'})
    
    resultado = ia.forward_chaining(['socrates es humano'])
    print(f"   Resultado: {resultado}")
    
    # 8. Chat con modelo de IA (requiere API key real)
    print("\n8. Probando chat con modelo de IA...")
    
    # Solo ejecutar si hay API key
    if ia.api_key and ia.api_key != 'tu_api_key_de_openrouter_aqui':
        respuesta = ia.chat_with_ai(
            "Explica qué es la IA en una oración",
            personality='profesor universitario'
        )
        print(f"   Pregunta: Explica qué es la IA en una oración")
        print(f"   Respuesta: {respuesta}")
    else:
        print("   Nota: Necesitas una API key real de OpenRouter para esta funcionalidad")
    
    # 9. Crear y ejecutar un agente
    print("\n9. Creando y ejecutando un agente...")
    
    # Definir funciones para el agente
    def percibir():
        return np.random.choice(['lluvia', 'sol', 'nublado'])
    
    def actuar(condicion):
        if condicion == 'lluvia':
            return 'llevar paraguas'
        elif condicion == 'sol':
            return 'llevar gafas de sol'
        else:
            return 'salir normalmente'
    
    # Crear agente simple
    agente_simple = ia.create_agent('agente_clima', 'simple', 
                                    perceive_fn=percibir, act_fn=actuar)
    
    # Ejecutar agente
    resultado_agente = ia.run_agent('agente_clima', steps=3)
    print(f"   Ejecución del agente: {resultado_agente}")
    
    # 10. Sistema de búsqueda
    print("\n10. Probando algoritmos de búsqueda...")
    grafo = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    
    camino_bfs = ia.bfs('A', 'F', grafo)
    print(f"   BFS de A a F: {camino_bfs}")
    
    # 11. Guardar estado
    print("\n11. Guardando estado del sistema...")
    estado_guardado = ia.save_state('estado_ia.pkl')
    print(f"   {estado_guardado}")
    
    # 12. Historial de conversación
    print("\n12. Historial de conversación:")
    historial = ia.get_conversation_history()
    for msg in historial:
        print(f"   [{msg.get('timestamp', '')[:19]}] {msg.get('role', '').title()}: {msg.get('message', '')[:50]}...")
    
    print("\n" + "="*60)
    print("✅ DEMOSTRACIÓN COMPLETADA")
    print("="*60)