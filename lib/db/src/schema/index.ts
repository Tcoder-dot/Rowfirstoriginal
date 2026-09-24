import { boolean, jsonb, pgEnum, pgTable, text, timestamp, uuid } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod/v4";

export const userRoleEnum = pgEnum("user_role", ["user", "researcher", "admin"]);

export const profiles = pgTable("profiles", {
  id: uuid("id").primaryKey(),
  email: text("email"),
  full_name: text("full_name"),
  avatar_url: text("avatar_url"),
  role: userRoleEnum("role").notNull().default("user"),
  is_active: boolean("is_active").notNull().default(true),
  created_at: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updated_at: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
});

export const analysisRuns = pgTable("analysis_runs", {
  id: uuid("id").primaryKey().defaultRandom(),
  user_id: uuid("user_id").notNull(),
  title: text("title").notNull(),
  status: text("status").notNull().default("queued"),
  source_type: text("source_type").default("upload"),
  metadata: jsonb("metadata").default({}),
  created_at: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updated_at: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
});

export const documents = pgTable("documents", {
  id: uuid("id").primaryKey().defaultRandom(),
  user_id: uuid("user_id").notNull(),
  analysis_run_id: uuid("analysis_run_id"),
  filename: text("filename").notNull(),
  storage_path: text("storage_path").notNull(),
  mime_type: text("mime_type").default("application/octet-stream"),
  created_at: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const insertProfileSchema = createInsertSchema(profiles).omit({
  created_at: true,
  updated_at: true,
});

export const insertAnalysisRunSchema = createInsertSchema(analysisRuns).omit({
  id: true,
  created_at: true,
  updated_at: true,
});

export const insertDocumentSchema = createInsertSchema(documents).omit({
  id: true,
  created_at: true,
});

export type Profile = typeof profiles.$inferSelect;
export type NewProfile = z.infer<typeof insertProfileSchema>;

export type AnalysisRun = typeof analysisRuns.$inferSelect;
export type NewAnalysisRun = z.infer<typeof insertAnalysisRunSchema>;

export type Document = typeof documents.$inferSelect;
export type NewDocument = z.infer<typeof insertDocumentSchema>;