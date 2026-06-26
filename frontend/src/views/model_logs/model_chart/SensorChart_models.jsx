import React, { useMemo } from "react";
import ReactECharts from "echarts-for-react";

export default function SensorChart({ measurements }) {
  // Build scatter series per sensor with per-point coloring 
  const series = useMemo(() => {
    const bySensor = {};

    (measurements || []).forEach(m => {
      const sensor_label = m.sensor_label ?? String(m.sensor_id ?? "unknown");
      if (!bySensor[sensor_label]) bySensor[sensor_label] = [];
      bySensor[sensor_label].push(m);
    });

    return Object.keys(bySensor).map(sensor_label => {
      const data = bySensor[sensor_label]
        .slice()
        .sort((a, b) => new Date(a.timestamp_utc) - new Date(b.timestamp_utc))
        .map(m => ({
          value: [new Date(m.timestamp_utc), m.value],
          itemStyle: { color: m.is_anomaly ? "#d32f2f" : "#9aa0a6" },
        }));

      return {
        name: sensor_label,
        type: "scatter",
        symbolSize: 6,
        emphasis: { focus: "series" },
        data,
      };
    });
  }, [measurements]);

  // Find min/max for y-axis scaling
  const { min, max } = useMemo(() => {
    if (!measurements?.length) return { min: 0, max: 1 };
    const values = measurements.map(m => m.value).filter(v => typeof v === "number" && !Number.isNaN(v));
    if (!values.length) return { min: 0, max: 1 };
    return { min: Math.min(...values), max: Math.max(...values) };
  }, [measurements]);

  const option = {
    grid: { left: 8, right: 16, top: 16, bottom: 88, containLabel: true },
    tooltip: { trigger: "axis", axisPointer: { type: "line" } },
    legend: { type: "scroll", bottom: 8, left: "center" },
    xAxis: {
      type: "time",
      boundaryGap: false,
      axisLabel: { hideOverlap: true },
    },
    yAxis: {
      type: "value",
      min,
      max,
      axisLabel: {
        align: "right",
        margin: 12,
      },
    },
    dataZoom: [{ type: "inside" }, { type: "slider", bottom: 32, height: 18 }],
    series,
  };

  return <ReactECharts option={option} style={{ height: "100%", width: "100%" }} />;
}