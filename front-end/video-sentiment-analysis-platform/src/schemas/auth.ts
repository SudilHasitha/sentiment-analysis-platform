import {z} from "zod"

export const loginSchema = z.object({
    email: z.string().email("Invalid email address"),
    password: z.string().min(8, "Password must have  8 chars")
});

export const signupSchema = z.object({
    name: z.string().min(2,"Length must be at least 2"),
    email: z.string().email("Invalid email"),
    password: z.string().min(8, "Password must have  8 chars"),
    confirmPassword :z.string(),
}).refine((data) => data.confirmPassword, {
    message: "Password do not match",
    path: ["confirmPassword"],
});

export type LoginSchema = z.infer<typeof loginSchema>;
export type signupSchema = z.infer<typeof signupSchema>;