-- AlterTable
ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN     "model_group_max_budget" JSONB;

-- AlterTable
ALTER TABLE "LiteLLM_UserTable" ADD COLUMN     "model_group_max_budget" JSONB NOT NULL DEFAULT '{}';
