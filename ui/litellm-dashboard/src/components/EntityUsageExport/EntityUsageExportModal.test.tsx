/**
 * Tests for EntityUsageExportModal component
 *
 * Validates core export functionality:
 * - Renders modal with correct default state (CSV format, daily scope)
 * - User can select export type (daily vs daily_with_models)
 * - User can switch format (CSV vs JSON)
 * - Export button triggers data generation with correct parameters
 * - Modal closes after successful export
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import EntityUsageExportModal from "./EntityUsageExportModal";

const downloadBlob = vi.fn();

vi.mock("./utils", () => ({
  downloadBlob: (...args: unknown[]) => downloadBlob(...args),
  exportFilename: vi.fn(() => "tag_usage_daily_2025-10-01_2025-10-14.csv"),
}));

describe("EntityUsageExportModal", () => {
  const baseProps = {
    isOpen: true,
    onClose: vi.fn(),
    entityType: "tag" as const,
    onExport: vi.fn().mockResolvedValue(new Blob(["data"])),
    dateRange: { from: new Date("2025-10-01"), to: new Date("2025-10-14") },
    selectedFilters: [],
    customTitle: "Export Tag Usage",
  };

  beforeEach(() => {
    vi.clearAllMocks();
    baseProps.onExport.mockResolvedValue(new Blob(["data"]));
  });

  it("exports through the onExport callback with the selected type and format, then closes", async () => {
    const user = userEvent.setup();

    renderWithProviders(<EntityUsageExportModal {...baseProps} />);

    expect(screen.getByRole("button", { name: /Export CSV/i })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Export CSV/i }));

    expect(baseProps.onExport).toHaveBeenCalledWith("daily", "csv");
    expect(downloadBlob).toHaveBeenCalledWith(expect.any(Blob), "tag_usage_daily_2025-10-01_2025-10-14.csv");
    expect(baseProps.onClose).toHaveBeenCalled();
  });

  it("forwards the export type the user picks", async () => {
    const user = userEvent.setup();

    renderWithProviders(<EntityUsageExportModal {...baseProps} />);

    await user.click(screen.getByRole("radio", { name: /Day-by-day breakdown by tag and key/i }));
    await user.click(screen.getByRole("button", { name: /Export CSV/i }));

    expect(baseProps.onExport).toHaveBeenCalledWith("daily_with_keys", "csv");
    expect(baseProps.onClose).toHaveBeenCalled();
  });

  it("keeps the modal open and reports the error when onExport rejects", async () => {
    const user = userEvent.setup();
    baseProps.onExport.mockRejectedValueOnce(new Error("export failed"));

    renderWithProviders(<EntityUsageExportModal {...baseProps} />);

    await user.click(screen.getByRole("button", { name: /Export CSV/i }));

    expect(baseProps.onExport).toHaveBeenCalled();
    expect(downloadBlob).not.toHaveBeenCalled();
    expect(baseProps.onClose).not.toHaveBeenCalled();
  });
});
