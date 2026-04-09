import { useCallback, useState } from 'react';
import DailyIframe from '@daily-co/daily-js';

interface Props {
  roomUrl: string;
  token: string;
  personaName: string;
  onEnd: () => void;
}

export default function CallInterface({ roomUrl, token, personaName, onEnd }: Props) {
  const [status, setStatus] = useState<'connecting' | 'connected' | 'ended'>('connecting');
  const [duration, setDuration] = useState(0);
  const [callFrame, setCallFrame] = useState<ReturnType<typeof DailyIframe.createCallObject> | null>(null);

  const startConnection = useCallback(async () => {
    const frame = DailyIframe.createCallObject({
      audioSource: true,
      videoSource: false,
    });

    frame.on('joined-meeting', () => {
      setStatus('connected');
      // 통화 시간 카운터
      const interval = setInterval(() => {
        setDuration((d) => d + 1);
      }, 1000);
      frame.on('left-meeting', () => {
        clearInterval(interval);
        setStatus('ended');
      });
    });

    frame.on('error', (e) => {
      console.error('Daily error:', e);
      setStatus('ended');
    });

    setCallFrame(frame);
    await frame.join({ url: roomUrl, token });
  }, [roomUrl, token]);

  // 자동 연결
  useState(() => {
    startConnection();
  });

  const handleEnd = async () => {
    if (callFrame) {
      await callFrame.leave();
      await callFrame.destroy();
    }
    onEnd();
  };

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-white rounded-xl shadow p-8 text-center space-y-6">
      {/* 상태 표시 */}
      <div className="space-y-2">
        <div
          className={`inline-block w-4 h-4 rounded-full ${
            status === 'connecting'
              ? 'bg-yellow-400 animate-pulse'
              : status === 'connected'
              ? 'bg-green-500 animate-pulse'
              : 'bg-gray-400'
          }`}
        />
        <p className="text-lg font-medium text-gray-900">
          {status === 'connecting' && '연결 중...'}
          {status === 'connected' && `${personaName}와 통화 중`}
          {status === 'ended' && '통화 종료'}
        </p>
      </div>

      {/* 통화 시간 */}
      <p className="text-4xl font-mono text-gray-700">{formatTime(duration)}</p>

      {/* 안내 */}
      {status === 'connected' && (
        <p className="text-sm text-gray-500">
          마이크로 말씀하세요. AI가 음성으로 응답합니다.
        </p>
      )}

      {/* 종료 버튼 */}
      {status !== 'ended' && (
        <button
          onClick={handleEnd}
          className="bg-red-600 text-white px-8 py-3 rounded-full text-lg font-medium hover:bg-red-700 transition"
        >
          통화 종료
        </button>
      )}

      {status === 'ended' && (
        <button
          onClick={onEnd}
          className="bg-gray-600 text-white px-8 py-3 rounded-full text-lg font-medium hover:bg-gray-700 transition"
        >
          돌아가기
        </button>
      )}
    </div>
  );
}
