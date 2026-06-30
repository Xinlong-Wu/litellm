import React, { useEffect, useRef, useState } from "react";
import { Button, Select, Space, Typography } from "antd";
import { DeleteOutlined, PlusOutlined } from "@ant-design/icons";
import { useAccessGroups, AccessGroupResponse } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";
import NumericalInput from "../shared/numerical_input";
import DurationSelect from "./DurationSelect";

const { Text } = Typography;

export interface ModelGroupBudgetConfig {
  max_budget?: number;
  budget_duration?: string;
}

/** Map of access_group_id -> budget config. */
export type ModelGroupBudgetValue = Record<string, ModelGroupBudgetConfig>;

interface EditorRow {
  groupId?: string;
  max_budget?: number;
  budget_duration?: string;
}

interface ModelGroupBudgetEditorProps {
  value?: ModelGroupBudgetValue;
  onChange?: (value: ModelGroupBudgetValue) => void;
}

const rowsToValue = (rows: EditorRow[]): ModelGroupBudgetValue =>
  rows.reduce<ModelGroupBudgetValue>((acc, row) => {
    if (row.groupId && row.max_budget !== undefined && row.max_budget !== null) {
      acc[row.groupId] = {
        max_budget: row.max_budget,
        ...(row.budget_duration ? { budget_duration: row.budget_duration } : {}),
      };
    }
    return acc;
  }, {});

const valueToRows = (value?: ModelGroupBudgetValue): EditorRow[] =>
  Object.entries(value ?? {}).map(([groupId, cfg]) => ({
    groupId,
    max_budget: cfg?.max_budget,
    budget_duration: cfg?.budget_duration,
  }));

/**
 * Editor for per-access-group dollar budgets.
 *
 * - Each row binds one access group (by access_group_id) to a max budget + reset window.
 * - Emits a `{ access_group_id: { max_budget, budget_duration } }` object via onChange,
 *   so it drops directly into an Ant Design `<Form.Item>`.
 */
const ModelGroupBudgetEditor: React.FC<ModelGroupBudgetEditorProps> = ({ value, onChange }) => {
  const { data: accessGroups, isLoading } = useAccessGroups();
  const [rows, setRows] = useState<EditorRow[]>(() => valueToRows(value));
  const lastEmitted = useRef<string>(JSON.stringify(value ?? {}));

  // Re-seed from an externally supplied value (e.g. form reset on edit load),
  // but ignore the echo of our own onChange to avoid a render loop.
  useEffect(() => {
    const incoming = JSON.stringify(value ?? {});
    if (incoming !== lastEmitted.current) {
      lastEmitted.current = incoming;
      setRows(valueToRows(value));
    }
  }, [value]);

  const emit = (next: EditorRow[]) => {
    setRows(next);
    const nextValue = rowsToValue(next);
    lastEmitted.current = JSON.stringify(nextValue);
    onChange?.(nextValue);
  };

  const updateRow = (index: number, patch: Partial<EditorRow>) =>
    emit(rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));

  const usedGroupIds = new Set(rows.map((r) => r.groupId).filter(Boolean) as string[]);

  const groupOptions = (accessGroups ?? []).map((group: AccessGroupResponse) => ({
    label: group.access_group_name,
    value: group.access_group_id,
  }));

  return (
    <div>
      {rows.length === 0 && (
        <Text className="text-xs text-gray-500 mb-2">No model group budgets set.</Text>
      )}
      {rows.map((row, index) => (
        <Space key={index} align="baseline" className="mb-2" style={{ display: "flex" }}>
          <Select
            showSearch
            placeholder="Access group"
            loading={isLoading}
            style={{ width: 220 }}
            value={row.groupId}
            optionFilterProp="label"
            onChange={(groupId) => updateRow(index, { groupId })}
            options={groupOptions.map((opt) => ({
              ...opt,
              disabled: opt.value !== row.groupId && usedGroupIds.has(opt.value),
            }))}
          />
          <NumericalInput
            step={0.01}
            min={0}
            placeholder="Max budget (USD)"
            style={{ width: 160 }}
            value={row.max_budget}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
              const raw = e.target.value;
              updateRow(index, { max_budget: raw === "" ? undefined : Number(raw) });
            }}
          />
          <DurationSelect
            value={row.budget_duration}
            onChange={(budget_duration) => updateRow(index, { budget_duration })}
          />
          <Button
            type="text"
            icon={<DeleteOutlined />}
            onClick={() => emit(rows.filter((_, i) => i !== index))}
          />
        </Space>
      ))}
      <Button
        type="dashed"
        icon={<PlusOutlined />}
        onClick={() => emit([...rows, {}])}
        style={{ width: "100%" }}
      >
        Add model group budget
      </Button>
    </div>
  );
};

export default ModelGroupBudgetEditor;
