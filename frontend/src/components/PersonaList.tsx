import { Persona } from '../api/client';

interface Props {
  personas: Persona[];
  onEdit: (persona: Persona) => void;
  onDelete: (id: string) => void;
}

export default function PersonaList({ personas, onEdit, onDelete }: Props) {
  if (personas.length === 0) {
    return <p className="text-gray-500">등록된 페르소나가 없습니다.</p>;
  }

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {personas.map((persona) => (
        <div
          key={persona.id}
          className="bg-white rounded-xl shadow p-5 space-y-3"
        >
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900">
              {persona.name}
            </h3>
            <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded">
              {persona.tts_provider}
            </span>
          </div>

          <p className="text-sm text-gray-600 line-clamp-2">
            {persona.greeting_message}
          </p>

          <div className="text-xs text-gray-400 space-y-1">
            <p>음성: {persona.tts_voice_id}</p>
            <p>모델: {persona.llm_model}</p>
            <p>
              VAD: {persona.vad_sensitivity} / 침묵감지:{' '}
              {persona.interrupt_threshold_ms}ms
            </p>
            {persona.knowledge_base_id && (
              <p>지식베이스: {persona.knowledge_base_id}</p>
            )}
          </div>

          <div className="flex gap-2 pt-2">
            <button
              onClick={() => onEdit(persona)}
              className="flex-1 text-sm bg-gray-100 text-gray-700 py-1.5 rounded-lg hover:bg-gray-200 transition"
            >
              수정
            </button>
            <button
              onClick={() => onDelete(persona.id)}
              className="flex-1 text-sm bg-red-50 text-red-600 py-1.5 rounded-lg hover:bg-red-100 transition"
            >
              삭제
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
