import { useEffect, useState } from 'react';
import type { Persona } from '../api/client';
import { getPersonas, deletePersona } from '../api/client';
import PersonaForm from '../components/PersonaForm';
import PersonaList from '../components/PersonaList';

export default function AdminDashboard() {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [editing, setEditing] = useState<Persona | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [loading, setLoading] = useState(true);

  const fetchPersonas = async () => {
    setLoading(true);
    try {
      const res = await getPersonas();
      setPersonas(res.data);
    } catch (err) {
      console.error('페르소나 목록 로드 실패:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPersonas();
  }, []);

  const handleDelete = async (id: string) => {
    if (!confirm('이 페르소나를 삭제하시겠습니까?')) return;
    await deletePersona(id);
    fetchPersonas();
  };

  const handleEdit = (persona: Persona) => {
    setEditing(persona);
    setShowForm(true);
  };

  const handleSaved = () => {
    setShowForm(false);
    setEditing(null);
    fetchPersonas();
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900">페르소나 관리</h2>
        <button
          onClick={() => { setEditing(null); setShowForm(!showForm); }}
          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition"
        >
          {showForm ? '닫기' : '+ 새 페르소나'}
        </button>
      </div>

      {showForm && (
        <div className="mb-8">
          <PersonaForm persona={editing} onSaved={handleSaved} />
        </div>
      )}

      {loading ? (
        <p className="text-gray-500">로딩 중...</p>
      ) : (
        <PersonaList
          personas={personas}
          onEdit={handleEdit}
          onDelete={handleDelete}
        />
      )}
    </div>
  );
}
