import React, { useState } from "react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import ModelGroupBudgetEditor, { ModelGroupBudgetValue } from "./ModelGroupBudgetEditor";

vi.mock("@/app/(dashboard)/hooks/accessGroups/useAccessGroups", () => ({
  useAccessGroups: vi.fn().mockReturnValue({
    data: [
      { access_group_id: "ag-1", access_group_name: "Premium" },
      { access_group_id: "ag-2", access_group_name: "Cheap" },
    ],
    isLoading: false,
    isError: false,
  }),
}));

function Harness({ initial }: { initial?: ModelGroupBudgetValue }) {
  const [value, setValue] = useState<ModelGroupBudgetValue | undefined>(initial);
  return (
    <>
      <ModelGroupBudgetEditor value={value} onChange={setValue} />
      <span data-testid="emitted">{JSON.stringify(value ?? {})}</span>
    </>
  );
}

describe("ModelGroupBudgetEditor", () => {
  it("renders a row per existing budget keyed by access_group_id", () => {
    render(<Harness initial={{ "ag-1": { max_budget: 50, budget_duration: "30d" } }} />);
    expect(screen.getByPlaceholderText("Access group")).toHaveValue("Premium");
  });

  it("adds a row and emits the keyed budget object on edit", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByText("Add model group budget"));

    const groupSelect = screen.getByPlaceholderText("Access group");
    await user.click(groupSelect);
    await user.click(await screen.findByText("Cheap"));

    const numberInput = screen.getByPlaceholderText("Max budget (USD)");
    fireEvent.change(numberInput, { target: { value: "12" } });

    const emitted = JSON.parse(screen.getByTestId("emitted").textContent || "{}");
    expect(emitted["ag-2"]).toBeDefined();
    expect(emitted["ag-2"].max_budget).toBe(12);
  });

  it("emits tpm_limit and rpm_limit alongside the budget", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByText("Add model group budget"));

    const groupSelect = screen.getByPlaceholderText("Access group");
    await user.click(groupSelect);
    await user.click(await screen.findByText("Premium"));

    fireEvent.change(screen.getByPlaceholderText("Max budget (USD)"), { target: { value: "12" } });
    fireEvent.change(screen.getByPlaceholderText("TPM limit"), { target: { value: "1000" } });
    fireEvent.change(screen.getByPlaceholderText("RPM limit"), { target: { value: "20" } });

    const emitted = JSON.parse(screen.getByTestId("emitted").textContent || "{}");
    expect(emitted["ag-1"]).toEqual({ max_budget: 12, tpm_limit: 1000, rpm_limit: 20 });
  });

  it("emits a rate-only row that has no budget", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByText("Add model group budget"));

    const groupSelect = screen.getByPlaceholderText("Access group");
    await user.click(groupSelect);
    await user.click(await screen.findByText("Cheap"));

    fireEvent.change(screen.getByPlaceholderText("RPM limit"), { target: { value: "5" } });

    const emitted = JSON.parse(screen.getByTestId("emitted").textContent || "{}");
    expect(emitted["ag-2"]).toEqual({ rpm_limit: 5 });
    expect(emitted["ag-2"].max_budget).toBeUndefined();
  });

  it("seeds tpm_limit and rpm_limit from an existing value", () => {
    render(<Harness initial={{ "ag-1": { max_budget: 50, tpm_limit: 800, rpm_limit: 10 } }} />);
    expect((screen.getByPlaceholderText("TPM limit") as HTMLInputElement).value).toBe("800");
    expect((screen.getByPlaceholderText("RPM limit") as HTMLInputElement).value).toBe("10");
  });

  it("removes a row and drops it from the emitted value", () => {
    render(<Harness initial={{ "ag-1": { max_budget: 50, budget_duration: "30d" } }} />);
    fireEvent.click(screen.getByRole("button", { name: "Remove model group budget 1" }));

    const emitted = JSON.parse(screen.getByTestId("emitted").textContent || "{}");
    expect(emitted["ag-1"]).toBeUndefined();
  });
});
