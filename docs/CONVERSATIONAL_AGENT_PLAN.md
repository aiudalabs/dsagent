# Plan de Transformación: DSAgent → Conversational Data Science Agent

## Análisis del Estado Actual

### Fortalezas del proyecto actual:
- **Arquitectura modular** con separación clara (agents/, core/, tools/, schema/)
- **Event streaming** ya implementado (perfecto para UI en tiempo real)
- **HITL Gateway** funcional (ya soporta interacción humana)
- **Jupyter kernel persistente** durante cada sesión
- **Sistema de logging estructurado** (eventos JSONL)
- **Integración MCP** para herramientas externas

### Limitaciones para conversacional:
1. **Sin persistencia de sesiones** - cada PlannerAgent es una ejecución única
2. **Kernel muere al terminar** - se pierde estado de variables/DataFrames
3. **Sin memoria a largo plazo** - no recuerda conversaciones previas
4. **Sin contexto de kernel** - el LLM no sabe qué variables existen
5. **CLI batch-oriented** - no diseñado para chat interactivo

---

## FASE 1: Persistencia de Sesiones y Conversaciones

**Estado:** [x] COMPLETADA (2026-01-07)

### 1.1 Session Manager
```
src/dsagent/session/
├── manager.py       # SessionManager (CRUD de sesiones)
├── store.py         # SessionStore (SQLite/JSON persistencia)
└── models.py        # Session, Message, KernelSnapshot
```

**Implementar:**
- `SessionManager` que maneja múltiples sesiones concurrentes
- `Session` como entidad con: id, created_at, messages[], kernel_state, plan_state
- Persistencia en SQLite (producción) o JSON (desarrollo)
- Restauración de sesión por ID

### 1.2 Conversation History
- Migrar de `self.messages: list` a `ConversationHistory` con:
  - Almacenamiento persistente
  - Métodos de búsqueda y filtrado
  - Truncamiento inteligente para contexto LLM
  - Sumarización automática de conversaciones largas

### 1.3 Message Threading
- Implementar modelo de mensajes rico:
```python
class ConversationMessage:
    id: str
    role: Literal["user", "assistant", "system", "execution"]
    content: str
    metadata: dict  # code_output, images, execution_time
    parent_id: Optional[str]  # Para threading
    timestamp: datetime
```

---

## FASE 2: Kernel State Management

**Estado:** [x] COMPLETADA (2026-01-07)

### 2.1 Kernel State Introspection
**Nuevo componente:** `KernelIntrospector`
```python
class KernelIntrospector:
    def get_defined_variables(self) -> dict[str, VariableInfo]
    def get_dataframe_summaries(self) -> dict[str, DataFrameSummary]
    def get_function_definitions(self) -> list[FunctionInfo]
    def get_imports(self) -> list[str]
    def get_execution_history(self) -> list[CellExecution]
```

**Ejecutar periódicamente:**
```python
# Inyectar en kernel para introspección
introspection_code = """
import json
_dsagent_state = {
    "variables": {k: type(v).__name__ for k,v in locals().items() if not k.startswith('_')},
    "dataframes": {k: {"shape": v.shape, "columns": list(v.columns)}
                   for k,v in locals().items() if isinstance(v, pd.DataFrame)}
}
print(json.dumps(_dsagent_state))
"""
```

### 2.2 Context Injection para LLM
- Antes de cada llamada LLM, inyectar contexto del kernel:
```xml
<kernel_context>
Variables definidas:
- df: DataFrame (1000 rows × 15 columns) - columns: [id, name, price, ...]
- model: RandomForestClassifier (fitted, accuracy=0.85)
- results: dict con 3 keys

Últimas 5 ejecuciones exitosas disponibles en historial.
</kernel_context>
```

### 2.3 Kernel Persistence (Avanzado)
- Serialización de estado con `dill` o `cloudpickle`
- Snapshot/restore del kernel
- Fallback: replay de código al restaurar sesión

---

## FASE 3: Interfaz Conversacional (CLI Interactivo)

**Estado:** [x] COMPLETADA (2026-01-07)

### 3.1 REPL Principal
**Nuevo archivo:** `src/dsagent/cli_interactive.py`

```python
class ConversationalCLI:
    def run(self):
        """REPL principal estilo Claude Code"""
        while True:
            user_input = self.prompt()  # Con Rich/Prompt Toolkit

            if user_input.startswith("/"):
                self.handle_command(user_input)
            else:
                for event in self.agent.chat(user_input):
                    self.render_event(event)
```

### 3.2 Comandos Slash
```
/new          - Nueva sesión (mantiene kernel opcional)
/sessions     - Listar sesiones previas
/load <id>    - Cargar sesión anterior
/context      - Mostrar estado del kernel
/vars         - Listar variables
/df <name>    - Inspeccionar DataFrame
/history      - Ver historial de código
/export       - Exportar notebook
/clear        - Limpiar pantalla
/undo         - Deshacer última ejecución
/help         - Ayuda
```

### 3.3 UI Enhancements
- Usar `rich` para renderizado de:
  - DataFrames (tablas formateadas)
  - Código con syntax highlighting
  - Imágenes inline (en terminales compatibles)
  - Progress bars para ejecuciones largas
  - Panel de estado del kernel

---

## FASE 4: Modo Conversacional del Agente

**Estado:** [x] COMPLETADA (2026-01-07)

### 4.1 Refactorizar AgentEngine
**Cambios en** `core/engine.py`:

```python
class AgentEngine:
    def __init__(self, session: Session):
        self.session = session
        self.kernel = session.kernel  # Kernel persistente

    async def chat(self, message: str) -> AsyncGenerator[AgentEvent]:
        """Modo conversacional - procesa un mensaje"""
        # 1. Añadir mensaje a historial
        self.session.add_message(role="user", content=message)

        # 2. Obtener contexto del kernel
        kernel_context = self.introspector.get_context()

        # 3. Construir prompt con contexto
        messages = self._build_messages(kernel_context)

        # 4. Streaming response
        async for event in self._process_response(messages):
            yield event
```

### 4.2 Nuevo System Prompt Conversacional
- Modificar `SYSTEM_PROMPT` para modo conversacional:
```python
CONVERSATIONAL_SYSTEM_PROMPT = """
You are a Data Science assistant in an interactive session.

## Current Session Context
{kernel_context}

## Conversation Guidelines
- You can reference variables and DataFrames that exist in the kernel
- Short tasks: execute directly without formal plan
- Complex tasks: create a plan, execute step by step
- You can ask clarifying questions before executing
- Remember previous conversation and executions

## Available in Kernel
{variables_summary}

## Recent Execution History
{execution_history}
"""
```

### 4.3 Intención del Usuario
**Nuevo componente:** `IntentClassifier`
```python
class IntentClassifier:
    def classify(self, message: str) -> Intent:
        """Determina tipo de solicitud"""
        # Tipos: QUESTION, EXECUTE_CODE, ANALYZE_DATA,
        #        EXPLAIN_RESULT, MODIFY_PREVIOUS, NEW_TASK
```

---

## FASE 5: Memoria y Contexto Largo

**Estado:** [x] COMPLETADA (2026-01-07) - Implementada Option A: Summarization

### 5.1 Memory Module
```
src/dsagent/memory/
├── __init__.py       # Exporta ConversationSummarizer, SummaryConfig, ConversationSummary
└── summarizer.py     # Sumarizador de conversaciones usando LLM
```

### 5.2 Sumarización Automática ✓
- Cuando conversación > N mensajes (configurable, default 30):
  - Sumarizar mensajes antiguos usando modelo LLM (gpt-4o-mini por defecto)
  - Mantener últimos K mensajes completos (configurable, default 10)
  - Almacenar sumario en ConversationHistory
  - Sumario inyectado automáticamente en contexto LLM
- `ConversationSummarizer`: Clase principal para sumarización
- `SummaryConfig`: Configuración (max_messages, keep_recent, model, temperature)
- `ConversationSummary`: Modelo para representar sumarios
- Fallback a sumario simple si LLM falla

### 5.3 Integración con ConversationalAgent ✓
- Config options: enable_summarization, summarization_threshold, keep_recent_messages, summarization_model
- Auto-summarize después de cada `chat()` y `chat_stream()`
- Sumarios incluyen: datos cargados, análisis realizados, modelos entrenados, hallazgos clave, estado del kernel

### 5.4 Retrieval Augmented Context (Futuro)
- Embeddings de mensajes/ejecuciones pasadas
- Recuperar contexto relevante para query actual
- Inyectar en prompt

---

## FASE 6: Notebook Vivo

**Estado:** [x] COMPLETADA (2026-01-07)

### 6.1 Live Notebook Builder
**Modificar** `utils/notebook.py`:
```python
class LiveNotebookBuilder:
    def __init__(self, path: Path):
        self.notebook_path = path

    def add_cell(self, cell: NotebookCell):
        """Añade celda y guarda inmediatamente"""
        self.cells.append(cell)
        self._save()

    def update_cell(self, index: int, cell: NotebookCell):
        """Actualiza celda existente"""

    def get_notebook(self) -> dict:
        """Retorna notebook completo"""
```

### 6.2 Sincronización Bidireccional
- El usuario puede editar el notebook en Jupyter
- Los cambios se reflejan en la sesión del agente
- Usar `watchdog` para detectar cambios

---

## FASE 7: Mejoras de Herramientas

**Estado:** [ ] Pendiente

### 7.1 Tool Discovery Dinámico
- El agente puede listar herramientas disponibles
- Sugerir herramientas relevantes según contexto

### 7.2 Data Source Tools
```python
# Nuevas herramientas integradas
tools = [
    "load_csv",
    "load_database",
    "fetch_api",
    "scrape_web",
    "load_from_cloud"  # S3, GCS, Azure Blob
]
```

### 7.3 Visualization Tools
- Generación interactiva de gráficos
- Templates de visualización
- Export a múltiples formatos

---

## FASE 8: API y Integraciones

**Estado:** [ ] Pendiente

### 8.1 WebSocket Server
**Nuevo:** `src/dsagent/server/websocket.py`
```python
@app.websocket("/chat/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    session = session_manager.get_or_create(session_id)
    while True:
        message = await websocket.receive_text()
        async for event in session.agent.chat(message):
            await websocket.send_json(event.model_dump())
```

### 8.2 REST API Enhancement
```
POST /sessions              # Crear sesión
GET  /sessions              # Listar sesiones
GET  /sessions/{id}         # Estado de sesión
POST /sessions/{id}/chat    # Enviar mensaje
GET  /sessions/{id}/kernel  # Estado del kernel
POST /sessions/{id}/execute # Ejecutar código directo
GET  /sessions/{id}/notebook # Exportar notebook
```

### 8.3 IDE Integration (Opcional)
- VS Code Extension
- Jupyter Lab Extension

---

## Orden de Implementación Sugerido

| Fase | Prioridad | Dependencias | Complejidad | Estado |
|------|-----------|--------------|-------------|--------|
| 1 - Session Persistence | Alta | - | Media | [x] COMPLETADA |
| 3 - CLI Interactivo | Alta | Fase 1 | Media | [x] COMPLETADA |
| 2 - Kernel State | Alta | Fase 1 | Alta | [x] COMPLETADA |
| 4 - Agent Conversacional | Alta | Fases 1-3 | Alta | [x] COMPLETADA |
| 6 - Notebook Vivo | Media | Fase 4 | Baja | [x] COMPLETADA |
| 5 - Memoria Larga | Media | Fase 4 | Alta | [x] COMPLETADA |
| 7 - Herramientas | Baja | Fase 4 | Media | [ ] |
| 8 - API/Integrations | Baja | Todas | Media | [ ] |

---

## Estructura de Archivos Propuesta

```
src/dsagent/
├── agents/
│   ├── base.py              # (mantener para compatibilidad)
│   └── conversational.py    # NUEVO: ConversationalAgent
├── core/
│   ├── engine.py            # (modificar para modo chat)
│   ├── executor.py          # (mantener)
│   ├── context.py           # (mantener)
│   ├── planner.py           # (mantener)
│   └── hitl.py              # (mantener)
├── session/                  # NUEVO
│   ├── __init__.py
│   ├── manager.py
│   ├── store.py
│   └── models.py
├── kernel/                   # NUEVO
│   ├── __init__.py
│   ├── introspector.py
│   ├── state.py
│   └── persistence.py
├── memory/                   # NUEVO
│   ├── __init__.py
│   ├── store.py
│   ├── summarizer.py
│   └── retriever.py
├── cli/                      # REFACTOR
│   ├── __init__.py
│   ├── commands.py
│   └── repl.py
├── server/                   # NUEVO
│   ├── __init__.py
│   ├── websocket.py
│   └── routes.py
└── schema/
    └── models.py            # (extender)
```

---

## Changelog

| Fecha | Fase | Cambios |
|-------|------|---------|
| 2026-01-07 | 5 | Fase 5 completada: Memoria Larga (Option A: Summarization) |
| - | - | - `memory/__init__.py`: Módulo de memoria con exports |
| - | - | - `memory/summarizer.py`: ConversationSummarizer, SummaryConfig, ConversationSummary |
| - | - | - `session/models.py`: Métodos de sumario en ConversationHistory (set_summary, apply_summary, etc.) |
| - | - | - `agents/conversational.py`: Integración de summarizer con auto-summarize |
| - | - | - `agents/conversational.py`: Config options para summarization (enable, threshold, model) |
| - | - | - `tests/test_summarization.py`: 25 tests para funcionalidad de summarization |
| - | - | - Sumarios inyectados automáticamente en contexto LLM |
| - | - | - Fallback a sumario simple si LLM falla |
| 2026-01-07 | 6 | Fase 6 completada: Notebook Vivo (Live Notebook) |
| - | - | - `utils/notebook.py`: LiveNotebookBuilder con auto-save atómico después de cada celda |
| - | - | - `utils/notebook.py`: JupyterFileWatcher usando watchdog para detectar cambios externos |
| - | - | - `utils/notebook.py`: LiveNotebookSync para sincronización bidireccional con Jupyter |
| - | - | - `utils/notebook.py`: NotebookChange modelo para representar cambios detectados |
| - | - | - `agents/conversational.py`: Config options enable_live_notebook y enable_notebook_sync |
| - | - | - `agents/conversational.py`: _create_notebook_builder() selecciona builder apropiado |
| - | - | - `agents/conversational.py`: get_live_notebook_path() para obtener path del notebook live |
| - | - | - `pyproject.toml`: watchdog>=3.0.0 agregado como dependencia |
| - | - | - `tests/test_live_notebook.py`: 33 tests para funcionalidad de live notebook |
| - | - | - Total: 371 tests (60 session + 58 cli + 34 kernel + 63 conversational + 33 live_notebook + 123 otros) |
| 2026-01-07 | 4.1 | Modo Híbrido implementado: Conversacional + Autónomo |
| - | - | - `agents/conversational.py`: Modo híbrido (sin plan → chat, con plan → autónomo) |
| - | - | - Extracción de `<plan>` con tracking de pasos [x]/[ ] |
| - | - | - Loop autónomo `_run_autonomous()` hasta plan completo o max_rounds |
| - | - | - `chat_stream()` para progreso en tiempo real |
| - | - | - Soporte HITL preparado (HITLGateway integrado) |
| - | - | - `cli/repl.py`: Muestra plan con progreso, indicador de rounds |
| - | - | - `tests/test_conversational.py`: 58 tests (plan, autónomo, stream) |
| 2026-01-07 | 4.4 | Integración de NotebookBuilder |
| - | - | - `agents/conversational.py`: NotebookBuilder integrado para auto-generar notebooks |
| - | - | - Trackeo de ejecuciones con step descriptions del plan |
| - | - | - `export_notebook()` genera notebook limpio con imports consolidados |
| - | - | - Auto-save en `shutdown()` y acceso via `/export` |
| - | - | - `cli/commands.py`: ExportCommand ahora funcional |
| - | - | - `cli/repl.py`: Agent expuesto en CLIContext |
| 2026-01-07 | 4.3 | System prompt mejorado para guardar outputs |
| - | - | - `agents/conversational.py`: Instrucciones explícitas para plt.savefig() |
| - | - | - Ejemplos de código para guardar plots, DataFrames y modelos |
| - | - | - Sección "CRITICAL: Saving Outputs" con patrones recomendados |
| 2026-01-07 | 4.2 | Comandos /data y /workspace mejorados |
| - | - | - `cli/commands.py`: DataCommand usa session.data_path (runs/{session_id}/data/) |
| - | - | - `cli/commands.py`: WorkspaceCommand usa session.workspace_path |
| - | - | - Fix: Archivos ahora se copian al workspace correcto de la sesión |
| - | - | - `tests/test_cli.py`: 8 tests actualizados para verificar paths de sesión |
| - | - | - Total: 210 tests (60 session + 58 cli + 34 kernel + 58 conversational) |
| 2026-01-07 | 4 | Fase 4 inicial: Agent Conversacional básico |
| - | - | - `agents/conversational.py`: ConversationalAgent + ChatResponse + ConversationalAgentConfig |
| - | - | - `cli/repl.py`: Integración completa con ConversationalAgent |
| - | - | - Sistema de prompts conversacional con kernel context injection |
| - | - | - Flujo de chat con ejecución de código y actualización de kernel snapshot |
| 2026-01-07 | 2 | Fase 2 completada: Kernel State Management implementado |
| - | - | - `kernel/backend.py`: ExecutorBackend interfaz abstracta + ExecutorConfig |
| - | - | - `kernel/introspector.py`: KernelIntrospector + IntrospectionResult |
| - | - | - `kernel/local.py`: LocalExecutor (implementa ExecutorBackend) |
| - | - | - `tests/test_kernel.py`: 34 tests pasando |
| - | - | - Arquitectura preparada para DockerExecutor/RemoteExecutor |
| 2026-01-07 | 3 | Fase 3 completada: CLI Interactivo implementado |
| - | - | - `cli/commands.py`: CommandRegistry + 13 comandos slash |
| - | - | - `cli/repl.py`: ConversationalCLI con prompt_toolkit |
| - | - | - `cli/renderer.py`: CLIRenderer con Rich |
| - | - | - `tests/test_cli.py`: 50 tests pasando |
| - | - | - Nuevo script: `dsagent-chat` |
| 2026-01-07 | 1 | Fase 1 completada: Session persistence implementada |
| - | - | - `session/models.py`: ConversationMessage, ConversationHistory, KernelSnapshot, Session |
| - | - | - `session/store.py`: JSONSessionStore, SQLiteSessionStore, SessionStore |
| - | - | - `session/manager.py`: SessionManager con CRUD completo |
| - | - | - `tests/test_session.py`: 60 tests pasando |
| - | - | Plan inicial creado |
