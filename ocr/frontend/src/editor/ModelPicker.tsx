import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { listModels, listSystemDevices } from "../api";

interface Props {
  family: string;
  device: string;
  onFamily: (f: string) => void;
  onDevice: (d: string) => void;
  compact?: boolean;
}

export function ModelPicker({ family, device, onFamily, onDevice, compact }: Props) {
  const modelsQ = useQuery({
    queryKey: ["models"],
    queryFn: listModels,
    staleTime: 0,
    retry: 1,
  });
  const devicesQ = useQuery({
    queryKey: ["devices"],
    queryFn: listSystemDevices,
    staleTime: 0,
    retry: 1,
  });

  const families = modelsQ.data?.families ?? [
    { family: "mock", available: true, supports_cuda: true, description: "", requires: [] },
  ];
  const allDevices = devicesQ.data?.devices ?? [
    { id: "cpu", kind: "cpu" as const, free_mb: null },
  ];

  // What does the currently selected family support?
  const currentFamily = families.find((f) => f.family === family);
  const supportsCuda = currentFamily?.supports_cuda ?? false;

  // Filter the device dropdown.
  const allowedDevices = allDevices.filter(
    (d) => d.kind === "cpu" || supportsCuda,
  );

  // If the current device is no longer allowed, snap to cpu — but only
  // once BOTH lists have really loaded. Previously this ran against the
  // placeholder lists while the requests were in flight, so a saved
  // "cuda:0" was reset to "cpu" on every mount.
  const ready = !!modelsQ.data && !!devicesQ.data;
  const allowedKey = allowedDevices.map((d) => d.id).join(",");
  useEffect(() => {
    if (!ready) return;
    const stillAllowed = allowedDevices.some((d) => d.id === device);
    if (!stillAllowed && device !== "cpu") onDevice("cpu");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, allowedKey, device]);

  // keep the saved device visible in the list while loading
  const deviceOptions = allowedDevices.some((d) => d.id === device)
    ? allowedDevices
    : [...allowedDevices, { id: device, kind: "cuda" as const, free_mb: null }];

  const size = compact ? "px-1 py-0.5 text-[10px]" : "px-2 py-1 text-[11px]";

  return (
    <div className="flex items-center gap-1">
      <select
        value={family}
        onChange={(e) => onFamily(e.target.value)}
        className={`rounded bg-slate-800 text-slate-200 ${size}`}
        title={
          currentFamily
            ? `${currentFamily.description}${
                currentFamily.supports_cuda ? " (CUDA-capable)" : " (CPU-only)"
              }`
            : "OCR adapter"
        }
      >
        {families.map((f) => (
          <option key={f.family} value={f.family} disabled={!f.available}>
            {f.family}
            {!f.available ? " (unavailable)" : ""}
            {f.available && !f.supports_cuda ? " · CPU" : ""}
          </option>
        ))}
      </select>

      <select
        value={device}
        onChange={(e) => onDevice(e.target.value)}
        className={`rounded bg-slate-800 text-slate-200 ${size}`}
        title="Device"
        disabled={ready && !supportsCuda}
      >
        {deviceOptions.map((d) => (
          <option key={d.id} value={d.id}>
            {d.id}
            {d.kind === "cuda" && d.free_mb != null
              ? ` (${Math.round(d.free_mb / 1024)} GB free)`
              : ""}
          </option>
        ))}
      </select>

      <button
        type="button"
        onClick={() => {
          modelsQ.refetch();
          devicesQ.refetch();
        }}
        className={`rounded bg-slate-800 text-slate-300 hover:bg-slate-700 ${size}`}
        title="Refresh model/device list"
      >
        ↻
      </button>

      {(modelsQ.error || devicesQ.error) && (
        <span
          className="text-[10px] text-rose-400"
          title={String(modelsQ.error ?? devicesQ.error)}
        >
          ⚠
        </span>
      )}
    </div>
  );
}