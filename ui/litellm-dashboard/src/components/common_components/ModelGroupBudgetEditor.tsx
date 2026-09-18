"use client";

import React, { useEffect, useRef, useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import { type AccessGroupResponse, useAccessGroups } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SearchSelect, type SearchSelectOption } from "../shared/SearchSelect";

export interface ModelGroupBudgetConfig {
  max_budget?: number;
  budget_duration?: string;
  tpm_limit?: number;
  rpm_limit?: number;
}

export type ModelGroupBudgetValue = Record<string, ModelGroupBudgetConfig>;

interface EditorRow {
  groupId?: string;
  max_budget?: number;
  budget_duration?: string;
  tpm_limit?: number;
  rpm_limit?: number;
}

interface ModelGroupBudgetEditorProps {
  value?: ModelGroupBudgetValue;
  onChange?: (value: ModelGroupBudgetValue) => void;
}

const DURATION_OPTIONS = [
  { label: "Daily", value: "24h" },
  { label: "Weekly", value: "7d" },
  { label: "Monthly", value: "30d" },
] as const;

const isSet = (value?: number): value is number => value !== undefined;

const rowsToValue = (rows: readonly EditorRow[]): ModelGroupBudgetValue =>
  Object.fromEntries(
    rows.flatMap((row) => {
      const hasLimit = isSet(row.max_budget) || isSet(row.tpm_limit) || isSet(row.rpm_limit);
      if (!row.groupId || !hasLimit) {
        return [];
      }

      const config: ModelGroupBudgetConfig = {
        ...(isSet(row.max_budget) ? { max_budget: row.max_budget } : {}),
        ...(row.budget_duration ? { budget_duration: row.budget_duration } : {}),
        ...(isSet(row.tpm_limit) ? { tpm_limit: row.tpm_limit } : {}),
        ...(isSet(row.rpm_limit) ? { rpm_limit: row.rpm_limit } : {}),
      };
      return [[row.groupId, config] as const];
    }),
  );

const valueToRows = (value?: ModelGroupBudgetValue): EditorRow[] =>
  Object.entries(value ?? {}).map(([groupId, cfg]) => ({
    groupId,
    max_budget: cfg?.max_budget,
    budget_duration: cfg?.budget_duration,
    tpm_limit: cfg?.tpm_limit,
    rpm_limit: cfg?.rpm_limit,
  }));

const ModelGroupBudgetEditor: React.FC<ModelGroupBudgetEditorProps> = ({ value, onChange }) => {
  const { data: accessGroups, isLoading } = useAccessGroups();
  const [rows, setRows] = useState<EditorRow[]>(() => valueToRows(value));
  const lastEmitted = useRef<string>(JSON.stringify(value ?? {}));

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

  const usedGroupIds = new Set(rows.flatMap((row) => (row.groupId ? [row.groupId] : [])));

  const groupOptions: SearchSelectOption[] = (accessGroups ?? []).map((group: AccessGroupResponse) => ({
    label: group.access_group_name,
    value: group.access_group_id,
  }));

  return (
    <div className="flex flex-col gap-3">
      {rows.length === 0 && <p className="text-xs text-muted-foreground">No model group budgets set.</p>}
      {rows.map((row, index) => (
        <div
          key={`${row.groupId ?? "new"}-${index}`}
          className="grid grid-cols-1 items-center gap-2 rounded-lg border border-border p-3 lg:grid-cols-[minmax(12rem,1.4fr)_repeat(4,minmax(8rem,1fr))_auto]"
        >
          <SearchSelect
            options={groupOptions.filter((option) => option.value === row.groupId || !usedGroupIds.has(option.value))}
            value={row.groupId}
            onValueChange={(groupId) => updateRow(index, { groupId: groupId || undefined })}
            placeholder="Access group"
            emptyText="No access groups"
            disabled={isLoading}
          />
          <Input
            type="number"
            step={0.01}
            min={0}
            placeholder="Max budget (USD)"
            aria-label={`Max budget ${index + 1}`}
            value={row.max_budget ?? ""}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
              const raw = e.target.value;
              updateRow(index, { max_budget: raw === "" ? undefined : Number(raw) });
            }}
          />
          <Select
            items={DURATION_OPTIONS}
            value={row.budget_duration ?? null}
            onValueChange={(budgetDuration: string | null) =>
              updateRow(index, { budget_duration: budgetDuration ?? undefined })
            }
          >
            <SelectTrigger aria-label={`Budget duration ${index + 1}`} className="w-full">
              <SelectValue placeholder="Reset window" />
            </SelectTrigger>
            <SelectContent>
              {DURATION_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Input
            type="number"
            step={1}
            min={0}
            placeholder="TPM limit"
            aria-label={`TPM limit ${index + 1}`}
            value={row.tpm_limit ?? ""}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
              const raw = e.target.value;
              updateRow(index, { tpm_limit: raw === "" ? undefined : Number(raw) });
            }}
          />
          <Input
            type="number"
            step={1}
            min={0}
            placeholder="RPM limit"
            aria-label={`RPM limit ${index + 1}`}
            value={row.rpm_limit ?? ""}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
              const raw = e.target.value;
              updateRow(index, { rpm_limit: raw === "" ? undefined : Number(raw) });
            }}
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label={`Remove model group budget ${index + 1}`}
            onClick={() => emit(rows.filter((_, rowIndex) => rowIndex !== index))}
          >
            <Trash2 />
          </Button>
        </div>
      ))}
      <Button type="button" variant="outline" className="w-full" onClick={() => emit([...rows, {}])}>
        <Plus />
        Add model group budget
      </Button>
    </div>
  );
};

export default ModelGroupBudgetEditor;
